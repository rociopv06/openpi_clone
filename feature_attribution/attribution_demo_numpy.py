#!/usr/bin/env python3
"""Pure-numpy feature attribution demo—no scipy, sklearn, pandas, or torch needed.
Trains a simple neural network from scratch and computes gradients.
"""
import numpy as np

class SimpleNN:
    """Minimal neural network with numpy."""
    def __init__(self, n_in, n_hidden=16):
        self.w1 = np.random.randn(n_in, n_hidden) * 0.01
        self.b1 = np.zeros(n_hidden)
        self.w2 = np.random.randn(n_hidden, 2) * 0.01
        self.b2 = np.zeros(2)
    
    def forward(self, x):
        """x: (batch, n_in). Returns (batch, 2) logits."""
        z1 = x @ self.w1 + self.b1
        a1 = np.maximum(0, z1)  # ReLU
        z2 = a1 @ self.w2 + self.b2
        return z2
    
    def softmax(self, logits):
        """logits: (batch, 2). Returns (batch, 2) probs."""
        e = np.exp(logits - logits.max(axis=1, keepdims=True))
        return e / e.sum(axis=1, keepdims=True)
    
    def compute_loss(self, x, y):
        """x: (batch, n_in), y: (batch,) binary labels."""
        logits = self.forward(x)
        probs = self.softmax(logits)
        batch_size = len(y)
        loss = -np.mean(np.log(probs[np.arange(batch_size), y] + 1e-7))
        return loss
    
    def gradient_input_class1(self, x):
        """Compute gradient of class-1 score w.r.t. inputs (for attribution).
        Returns (batch, n_in) gradient matrix.
        """
        eps = 1e-5
        grad = np.zeros_like(x)
        baseline_score = self.forward(x)[:, 1].mean()
        for i in range(x.shape[1]):
            x_pert = x.copy()
            x_pert[:, i] += eps
            pert_score = self.forward(x_pert)[:, 1].mean()
            grad[:, i] = (pert_score - baseline_score) / eps
        return grad
    
    def train_step(self, x, y, lr=0.1):
        """Simple gradient descent update."""
        batch_size = len(y)
        # Forward pass
        z1 = x @ self.w1 + self.b1
        a1 = np.maximum(0, z1)
        z2 = a1 @ self.w2 + self.b2
        logits = z2
        
        # Backward pass
        probs = self.softmax(logits)
        probs[np.arange(batch_size), y] -= 1
        probs /= batch_size
        
        dz2 = probs
        dw2 = a1.T @ dz2
        db2 = dz2.sum(axis=0)
        
        da1 = dz2 @ self.w2.T
        da1[z1 <= 0] = 0  # ReLU gradient
        dw1 = x.T @ da1
        db1 = da1.sum(axis=0)
        
        # Update
        self.w1 -= lr * dw1
        self.b1 -= lr * db1
        self.w2 -= lr * dw2
        self.b2 -= lr * db2

def main():
    print("Feature Attribution Demo (Pure NumPy)")
    print("=" * 50)
    
    # Generate synthetic data
    np.random.seed(42)
    n_samples, n_features = 1000, 8
    X = np.random.randn(n_samples, n_features)
    # Synthetic target: class 1 if f0 + f1 + 0.5*f2 > 0, else 0
    y = (X[:, 0] + X[:, 1] + 0.5*X[:, 2] > 0).astype(int)
    
    # Split
    split = int(0.7 * n_samples)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]
    
    # Normalize
    mean, std = X_train.mean(axis=0), X_train.std(axis=0)
    X_train = (X_train - mean) / (std + 1e-7)
    X_test = (X_test - mean) / (std + 1e-7)
    
    # Train model
    print("Training model...")
    model = SimpleNN(n_features, n_hidden=16)
    for epoch in range(100):
        idx = np.random.permutation(len(X_train))
        for i in range(0, len(X_train), 32):
            batch_idx = idx[i:i+32]
            model.train_step(X_train[batch_idx], y_train[batch_idx], lr=0.01)
        if (epoch + 1) % 25 == 0:
            loss = model.compute_loss(X_train, y_train)
            print(f"  Epoch {epoch+1}: loss = {loss:.4f}")
    
    # Evaluate
    logits = model.forward(X_test)
    preds = logits.argmax(axis=1)
    acc = (preds == y_test).mean()
    print(f"\nTest Accuracy: {acc:.4f}")
    
    # Feature attribution: input gradient
    print("\nComputing input gradients for feature attribution...")
    grads = model.gradient_input_class1(X_test)
    attr_scores = np.mean(np.abs(grads), axis=0)
    
    print("\nFeature Attribution Scores (mean absolute gradient):")
    for i, score in enumerate(attr_scores):
        print(f"  Feature {i}: {score:.6f}")
    
    print("\nTop 3 most important features:")
    top_idx = np.argsort(-attr_scores)[:3]
    for rank, i in enumerate(top_idx, 1):
        print(f"  {rank}. Feature {i}: {attr_scores[i]:.6f}")
    
    print("\n✓ Demo completed successfully!")

if __name__ == "__main__":
    main()
