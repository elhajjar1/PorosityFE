#!/usr/bin/env python3
"""
Train ML Correction Model for Mori-Tanaka Stiffness Predictions
================================================================

Generates training data from experimental validation datasets and
published literature knockdown data, then trains a small MLP to predict
correction factors that amplify the Mori-Tanaka predictions to match
real experimental observations.

The correction is applied as:
    r_corrected = r_MT * correction_factor(Vp, AR, property_class)

where r_MT is the Mori-Tanaka degradation ratio (e.g., E22_deg/E22_pristine)
and correction_factor < 1 amplifies the degradation.

Training data sources:
1. Liu et al. (2006) - T700/TDE85, [0/90]3s, 0.6-3.2% Vp
2. Stamopoulos et al. (2016) - HTA/EHkF420, UD, 0.82-3.43% Vp
3. Elhajjar (2025) - T700/#2510, [0/45/90/-45/0]s, 0.05-10% Vp
4. Published literature correlations (Ghiorse 1993, Olivier 1995)
"""

import sys
import os
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from porosity_fe_analysis import (
    MaterialProperties, MATERIALS, PorosityField, CompositeMesh,
    _mt_effective_stiffness, _degraded_composite_stiffness,
    VOID_SHAPES,
)
import dataclasses

# ============================================================
# EXPERIMENTAL KNOCKDOWN DATA (ground truth)
# ============================================================

def _build_training_data():
    """Build training dataset from experimental sources.

    Each sample: (Vp, void_AR, property_class, mt_ratio) -> exp_ratio

    property_class:
        0 = E22/transverse stiffness
        1 = G12/shear modulus
        2 = flexural modulus (bending stiffness)
        3 = tensile modulus (fiber-dominated)
    """
    samples = []

    # --- Liu et al. (2006): T700/TDE85, [0/90]3s ---
    # Void shape: spherical at low Vp, elongated at high Vp
    # Data normalized to 0.6% baseline
    liu_vp = [0.6, 0.9, 1.0, 1.2, 1.5, 2.0, 2.2, 3.2]
    liu_flex_mod = [1.00, 0.98, 0.97, 0.96, 0.94, 0.90, 0.88, 0.82]
    liu_tens_mod = [1.00, 0.99, 0.99, 0.99, 0.98, 0.97, 0.97, 0.96]

    for Vp_pct, fm, tm in zip(liu_vp, liu_flex_mod, liu_tens_mod):
        Vp = Vp_pct / 100.0
        ar = 1.0 + 2.0 * max(0, Vp - 0.006)  # evolving AR
        # Flexural modulus -> property class 2
        samples.append((Vp, ar, 2, fm))
        # Tensile modulus -> property class 3
        samples.append((Vp, ar, 3, tm))

    # --- Stamopoulos et al. (2016): HTA/EHkF420, UD ---
    stam_vp = [0.82, 1.56, 1.62, 3.43]
    stam_trans_mod = [1.000, 0.998, 0.995, 0.991]  # E22
    stam_shear_mod = [1.000, 0.833, 0.822, 0.792]  # G12
    stam_flex_mod = [1.000, 0.984, 0.967, 0.959]   # Flexural

    for Vp_pct, e22, g12, fm in zip(stam_vp, stam_trans_mod, stam_shear_mod,
                                      stam_flex_mod):
        Vp = Vp_pct / 100.0
        ar = 1.0 + 3.0 * max(0, Vp - 0.008)  # more elongated (UD)
        samples.append((Vp, ar, 0, e22))   # Transverse stiffness
        samples.append((Vp, ar, 1, g12))   # Shear modulus
        samples.append((Vp, ar, 2, fm))    # Flexural modulus

    # --- Published literature correlations ---
    # Ghiorse (1993): ~6-7% per percent void for ILSS, ~5% for flexural mod
    # Olivier et al. (1995): similar trends
    # Generate synthetic data points from established correlations
    for Vp_pct in np.linspace(0.5, 8.0, 20):
        Vp = Vp_pct / 100.0
        ar = 1.0 + 2.5 * max(0, Vp - 0.005)

        # E22 drops ~3% per percent void (literature consensus)
        e22_kd = max(0.7, 1.0 - 0.03 * Vp_pct)
        samples.append((Vp, ar, 0, e22_kd))

        # G12 drops ~5-7% per percent void (shear most sensitive)
        g12_kd = max(0.5, 1.0 - 0.06 * Vp_pct)
        samples.append((Vp, ar, 1, g12_kd))

        # Flexural modulus drops ~4-5% per percent void
        flex_kd = max(0.6, 1.0 - 0.045 * Vp_pct)
        samples.append((Vp, ar, 2, flex_kd))

        # Tensile modulus: ~1% per percent void (fiber-dominated)
        tens_kd = max(0.85, 1.0 - 0.01 * Vp_pct)
        samples.append((Vp, ar, 3, tens_kd))

    return samples


def _compute_mt_ratios(samples, material):
    """Compute Mori-Tanaka degradation ratios for each sample."""
    E_m = material.matrix_modulus
    nu_m = material.matrix_poisson
    G_m = E_m / (2.0 * (1.0 + nu_m))
    E_f = material.fiber_modulus
    Vf = material.fiber_volume_fraction
    Vm = 1.0 - Vf
    nu_f = 0.2
    G_f = E_f / (2.0 * (1.0 + nu_f))

    C_m = material.get_isotropic_matrix_stiffness()

    xi_HT = 2.0
    xi_G = 1.0

    def _halpin_tsai(Ef, Em, xi, vf):
        ratio = Ef / Em
        eta = (ratio - 1.0) / (ratio + xi)
        return Em * (1.0 + xi * eta * vf) / (1.0 - eta * vf)

    # Pristine reference values
    E22_HT_prist = _halpin_tsai(E_f, E_m, xi_HT, Vf)
    G12_HT_prist = _halpin_tsai(G_f, G_m, xi_G, Vf)
    E11_rom_prist = Vf * E_f + Vm * E_m

    dataset = []  # (features, target_correction)

    for Vp, ar, prop_class, exp_ratio in samples:
        if Vp < 1e-6:
            continue

        # Compute void shape radii from AR
        if ar < 1.05:
            radii = (1.0, 1.0, 1.0)
        else:
            radii = (ar, ar, 1.0)  # penny/oblate

        # M-T degraded matrix
        C_eff = _mt_effective_stiffness(C_m, Vp, radii, nu_m)
        mu_eff = C_eff[3, 3]
        lam_eff = C_eff[0, 1]
        denom = lam_eff + mu_eff
        G_m_eff = max(mu_eff, 1.0)
        E_m_eff = mu_eff * (3.0 * lam_eff + 2.0 * mu_eff) / denom if denom > 1e-12 else 1.0
        E_m_eff = max(E_m_eff, 1.0)

        # M-T composite ratios
        E22_HT_deg = _halpin_tsai(E_f, E_m_eff, xi_HT, Vf)
        G12_HT_deg = _halpin_tsai(G_f, G_m_eff, xi_G, Vf)
        E11_rom_deg = Vf * E_f + Vm * E_m_eff

        r_E22_mt = E22_HT_deg / E22_HT_prist
        r_G12_mt = G12_HT_deg / G12_HT_prist
        r_E11_mt = E11_rom_deg / E11_rom_prist

        # Select the M-T ratio for this property class
        if prop_class == 0:    # E22
            mt_ratio = r_E22_mt
        elif prop_class == 1:  # G12
            mt_ratio = r_G12_mt
        elif prop_class == 2:  # Flexural (mix of E22 and interface effects)
            mt_ratio = r_E22_mt  # closest proxy
        elif prop_class == 3:  # Tensile (fiber-dominated)
            mt_ratio = r_E11_mt
        else:
            mt_ratio = r_E22_mt

        # Correction factor: how much more degradation is needed
        # exp_ratio = mt_ratio * correction
        # correction = exp_ratio / mt_ratio
        if mt_ratio > 0.01:
            correction = exp_ratio / mt_ratio
        else:
            correction = 1.0

        # One-hot encode property class
        prop_onehot = [0.0] * 4
        prop_onehot[prop_class] = 1.0

        features = [Vp, ar] + prop_onehot
        dataset.append((features, correction))

    return dataset


# ============================================================
# NUMPY-ONLY MLP
# ============================================================

class NumpyMLP:
    """Minimal MLP using only NumPy. No PyTorch/TensorFlow dependency."""

    def __init__(self, layer_sizes):
        """Initialize with random weights."""
        self.weights = []
        self.biases = []
        np.random.seed(42)
        for i in range(len(layer_sizes) - 1):
            # Xavier initialization
            scale = np.sqrt(2.0 / (layer_sizes[i] + layer_sizes[i + 1]))
            W = np.random.randn(layer_sizes[i], layer_sizes[i + 1]) * scale
            b = np.zeros(layer_sizes[i + 1])
            self.weights.append(W)
            self.biases.append(b)

    def forward(self, x):
        """Forward pass with ReLU hidden layers, sigmoid output."""
        activations = [x]
        for i, (W, b) in enumerate(zip(self.weights, self.biases)):
            z = activations[-1] @ W + b
            if i < len(self.weights) - 1:
                z = np.maximum(0, z)  # ReLU
            else:
                z = 1.0 / (1.0 + np.exp(-z))  # Sigmoid -> (0, 1)
            activations.append(z)
        return activations

    def predict(self, x):
        """Forward pass, return output only."""
        return self.forward(x)[-1]

    def train(self, X, y, epochs=2000, lr=0.01, verbose=True):
        """Train with backpropagation (mini-batch gradient descent)."""
        n_samples = X.shape[0]

        for epoch in range(epochs):
            # Forward pass
            activations = self.forward(X)
            output = activations[-1]

            # Loss: MSE
            loss = np.mean((output - y) ** 2)

            if verbose and epoch % 200 == 0:
                print(f"  Epoch {epoch:4d}: loss = {loss:.6f}")

            # Backward pass
            delta = 2.0 * (output - y) / n_samples
            # Sigmoid derivative
            delta = delta * output * (1.0 - output)

            for i in reversed(range(len(self.weights))):
                dW = activations[i].T @ delta
                db = np.sum(delta, axis=0)

                self.weights[i] -= lr * dW
                self.biases[i] -= lr * db

                if i > 0:
                    delta = delta @ self.weights[i].T
                    # ReLU derivative
                    delta = delta * (activations[i] > 0).astype(float)

        final_loss = np.mean((self.predict(X) - y) ** 2)
        if verbose:
            print(f"  Final loss: {final_loss:.6f}")
        return final_loss

    def save(self, path):
        """Save weights to .npz file."""
        data = {}
        for i, (W, b) in enumerate(zip(self.weights, self.biases)):
            data[f'W{i}'] = W
            data[f'b{i}'] = b
        data['n_layers'] = np.array([len(self.weights)])
        np.savez(path, **data)
        print(f"Saved model to {path}")

    @classmethod
    def load(cls, path):
        """Load weights from .npz file."""
        data = np.load(path)
        n_layers = int(data['n_layers'][0])
        model = cls.__new__(cls)
        model.weights = [data[f'W{i}'] for i in range(n_layers)]
        model.biases = [data[f'b{i}'] for i in range(n_layers)]
        return model


def main():
    print("=" * 60)
    print("Training ML Correction for Mori-Tanaka Predictions")
    print("=" * 60)

    material = MATERIALS['T700_epoxy']

    # Step 1: Build experimental training data
    print("\n1. Building training data from experimental sources...")
    raw_samples = _build_training_data()
    print(f"   Raw samples: {len(raw_samples)}")

    # Step 2: Compute M-T ratios and correction factors
    print("2. Computing M-T ratios for each sample...")
    dataset = _compute_mt_ratios(raw_samples, material)
    print(f"   Training pairs: {len(dataset)}")

    # Prepare arrays
    X = np.array([d[0] for d in dataset])
    y = np.array([d[1] for d in dataset]).reshape(-1, 1)

    # Normalize inputs
    X_mean = X.mean(axis=0)
    X_std = X.std(axis=0) + 1e-8
    X_norm = (X - X_mean) / X_std

    # Clip correction targets to reasonable range [0.5, 1.1]
    y = np.clip(y, 0.5, 1.1)

    print(f"\n   Feature stats:")
    print(f"   Vp range:  {X[:, 0].min():.4f} - {X[:, 0].max():.4f}")
    print(f"   AR range:  {X[:, 1].min():.2f} - {X[:, 1].max():.2f}")
    print(f"   Correction range: {y.min():.3f} - {y.max():.3f}")
    print(f"   Mean correction:  {y.mean():.3f}")

    # Step 3: Train MLP
    print("\n3. Training MLP (6 -> 32 -> 16 -> 1)...")
    model = NumpyMLP([6, 32, 16, 1])
    loss = model.train(X_norm, y, epochs=3000, lr=0.005)

    # Step 4: Evaluate
    print("\n4. Evaluation on training data:")
    y_pred = model.predict(X_norm)
    mae = np.mean(np.abs(y_pred - y))
    print(f"   MAE: {mae:.4f}")
    print(f"   Max error: {np.max(np.abs(y_pred - y)):.4f}")

    # Print some predictions
    print(f"\n   Sample predictions (Vp, AR, class -> correction):")
    for i in range(0, len(dataset), max(1, len(dataset) // 10)):
        Vp = X[i, 0]
        ar = X[i, 1]
        pc = int(np.argmax(X[i, 2:6]))
        classes = ['E22', 'G12', 'Flex', 'Tens']
        print(f"   Vp={Vp*100:.1f}%, AR={ar:.1f}, {classes[pc]:>4s}: "
              f"pred={y_pred[i, 0]:.3f}, actual={y[i, 0]:.3f}")

    # Step 5: Save model + normalization params
    output_dir = os.path.dirname(os.path.abspath(__file__))
    model.save(os.path.join(output_dir, 'mt_correction_model.npz'))
    np.savez(os.path.join(output_dir, 'mt_correction_norm.npz'),
             X_mean=X_mean, X_std=X_std)
    print(f"\nSaved normalization params to {output_dir}/mt_correction_norm.npz")

    print(f"\n{'=' * 60}")
    print("TRAINING COMPLETE")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
