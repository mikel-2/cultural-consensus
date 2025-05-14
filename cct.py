# cct.py

import pandas as pd
import numpy as np
import pymc as pm
import aesara.tensor as at
import arviz as az

def load_plant_knowledge_data(path="data/plant_knowledge.csv"):
    df = pd.read_csv(path)
    df = df.drop(columns=['Informant'])
    return df.values

# Load data
X = load_plant_knowledge_data()
N, M = X.shape

# Model
with pm.Model() as cct_model:
    D = pm.Uniform("D", lower=0.5, upper=1.0, shape=N)
    Z = pm.Bernoulli("Z", p=0.5, shape=M)

    D_reshaped = D[:, None]
    Z_reshaped = Z[None, :]

    p = Z_reshaped * D_reshaped + (1 - Z_reshaped) * (1 - D_reshaped)
    X_obs = pm.Bernoulli("X_obs", p=p, observed=X)

    trace = pm.sample(draws=2000, chains=4, tune=1000, target_accept=0.9, random_seed=42)

# Analysis
az.summary(trace, var_names=["D", "Z"])
az.plot_posterior(trace, var_names=["D"])
az.plot_posterior(trace, var_names=["Z"])

D_mean = trace.posterior["D"].mean(dim=["chain", "draw"]).values
Z_mean = trace.posterior["Z"].mean(dim=["chain", "draw"]).values
Z_mode = (Z_mean > 0.5).astype(int)

most_competent = np.argmax(D_mean)
least_competent = np.argmin(D_mean)

majority_vote = (X.mean(axis=0) > 0.5).astype(int)
agreement = np.mean(majority_vote == Z_mode)
print(f"Consensus agreement with majority vote: {agreement * 100:.1f}%")
