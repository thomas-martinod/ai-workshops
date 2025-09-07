import torch

class Net:
    def __init__(self): self.layers=[]; self.training=True
    def add(self, layer): self.layers.append(layer)
    def train(self): self.training=True; [getattr(l,'train',lambda:None)() for l in self.layers]; return self
    def eval(self):  self.training=False;[getattr(l,'eval', lambda:None)() for l in self.layers]; return self
    def forward(self, X):
        for layer in self.layers: X = layer.forward(X)
        return X
    def backward(self, dZ):
        for layer in reversed(self.layers): dZ = layer.backward(dZ)
        return dZ
    def update(self, lr):
        for layer in self.layers:
            if hasattr(layer, "update"): layer.update(lr)




class Linear:
    """
    A simple fully connected (dense) layer.
    Performs a linear transformation:  Z = XW + b
    """

    def __init__(self, nin, nout, device="cpu"):
        """
        Initialize the layer parameters.
        """
        # Initialize weights from a normal distribution
        self.W = torch.randn(nin, nout, device=device, requires_grad=False)
        # Initialize biases to zero
        self.b = torch.zeros(nout, device=device, requires_grad=False)
        self.training = True  # for compatibility with Dropout/BatchNorm

    def train(self):
        """Switch to training mode."""
        self.training = True
        return self

    def eval(self):
        """Switch to evaluation mode."""
        self.training = False
        return self

    def forward(self, X):
        """
        Forward pass: compute the output of the layer.
        """
        self.X = X
        Z = X @ self.W + self.b
        return Z

    def backward(self, dZ):
        """
        Backward pass: compute gradients w.r.t. W, b, and X.
        """
        # Gradients
        dW = self.X.T @ dZ                   # shape (nin, nout)
        db = torch.sum(dZ, dim=0)            # shape (nout,), check if torch.sum or np.sum
        dX = dZ @ self.W.T                   # shape (batch, nin)

        # Store for update step
        self.dW = dW
        self.db = db
        return dX

    def update(self, lr):
        """
        Update parameters using gradient descent.
        """
        self.W -= lr * self.dW
        self.b -= lr * self.db





class ReLU:
    """
    ReLU activation layer.
    """

    def forward(self, Z):
        """
        Perform the forward pass of the ReLU activation function.

        Args:
            Z (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor with ReLU applied element-wise.
        """
        self.Z = Z  # 🔑 guardar para el backward
        self.A = torch.clamp(Z, min=0)  # equivalente a max(0, Z)
        return self.A

    def backward(self, dA):
        """
        Perform the backward pass of the ReLU activation function.

        Args:
            dA (torch.Tensor): Gradient of the loss with respect to the output.

        Returns:
            dZ torch.Tensor: Gradient of the loss with respect to the input.
        """
        dZ = dA.clone()  # copiar para no modificar dA
        dZ[self.Z <= 0] = 0  # 🔑 gradiente se bloquea en las entradas negativas
        return dZ

    def update(self, lr):
        """
        ReLU does not have any parameters to update.
        """
        pass




class BatchNorm1D:
    """
    Batch Normalization for 2D inputs: (batch, features).
    """

    def __init__(self, n_features, eps=1e-5, momentum=0.1, device="cpu"):
        self.eps = eps
        self.momentum = momentum
        self.device = device

        # Learnable affine parameters
        self.gamma = torch.ones(n_features, device=device, requires_grad=False)
        self.beta  = torch.zeros(n_features, device=device, requires_grad=False)

        # Running (inference) statistics
        self.running_mean = torch.zeros(n_features, device=device, requires_grad=False)
        self.running_var  = torch.ones(n_features,  device=device, requires_grad=False)

        # Mode flag
        self.training = True

        # Caches for backward (reset each forward)
        self.reset_cache()

        # Grads for parameters
        self.dgamma = None
        self.dbeta  = None

    def reset_cache(self):
        """Clear cached values (called at start of each forward)."""
        self.X = None
        self.X_hat = None
        self.batch_mean = None
        self.batch_var = None
        self.std = None

    def train(self): 
        self.training = True
        return self

    def eval(self):  
        self.training = False
        return self

    def forward(self, X):
        # reset cache at each forward
        self.reset_cache()

        if self.training:
            # ===== batch statistics =====
            self.batch_mean = X.mean(dim=0)
            self.batch_var  = X.var(dim=0, unbiased=False)

            # ===== normalize =====
            self.std = torch.sqrt(self.batch_var + self.eps)
            self.X_hat = (X - self.batch_mean) / self.std

            # ===== update running stats =====
            self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * self.batch_mean
            self.running_var  = (1 - self.momentum) * self.running_var  + self.momentum * self.batch_var
        else:
            # ===== inference normalization =====
            self.std = torch.sqrt(self.running_var + self.eps)
            self.X_hat = (X - self.running_mean) / self.std

        # Cache input
        self.X = X

        # ===== affine transform =====
        Y = self.gamma * self.X_hat + self.beta
        return Y

    def backward(self, dY):
        if not self.training:
            raise RuntimeError("Backward called in eval() mode. Use training mode for gradient computation.")
        if self.X is None or self.std is None:
            raise RuntimeError("Cache is empty. Did you forget to call forward() before backward()?")

        m = dY.size(0)

        # ===== parameter gradients =====
        self.dbeta  = dY.sum(dim=0)
        self.dgamma = torch.sum(dY * self.X_hat, dim=0)

        # ===== gradient wrt normalized activations =====
        dx_hat = dY * self.gamma

        # Cached values
        x_mu   = self.X - self.batch_mean
        invstd = 1.0 / self.std  # safe because std is cached

        # ===== gradients of mean and variance =====
        dvar  = torch.sum(dx_hat * x_mu * -0.5 * (invstd ** 3), dim=0)
        dmean = torch.sum(-dx_hat * invstd, dim=0) + dvar * torch.mean(-2.0 * x_mu, dim=0)

        # ===== gradient wrt input =====
        dX = dx_hat * invstd + (2.0 / m) * x_mu * dvar + dmean / m

        return dX

    def update(self, lr):
        # Simple SGD step
        if self.dgamma is None or self.dbeta is None:
            raise RuntimeError("No gradients available. Did you forget to call backward()?")

        self.gamma -= lr * self.dgamma
        self.beta  -= lr * self.dbeta



class Dropout:
    """
    Inverted Dropout (for fully-connected tensors [batch, features]).

    - TRAIN: randomly zeroes activations with prob p, and rescales by 1/(1-p)
             so the expected activation stays constant.
    - EVAL:  identity (no dropout, no scaling).
    """

    def __init__(self, p=0.5, device="cpu"):
        assert 0.0 <= p < 1.0, "p must be in [0, 1)."
        self.p = p
        self.device = device
        self.training = True
        self.mask = None  # cache for backward

    # Mode control
    def train(self):
        self.training = True
        return self

    def eval(self):
        self.training = False
        return self

    def forward(self, X):
        if self.training and self.p > 0.0:
            # ===== Inverted dropout mask =====
            keep_prob = 1.0 - self.p
            # Bernoulli mask {0,1} with prob keep_prob
            bernoulli = torch.bernoulli(torch.full_like(X, keep_prob, device=self.device))
            # Scale mask to {0, 1/keep_prob}
            self.mask = bernoulli / keep_prob
            # Apply mask
            return X * self.mask
        else:
            # In eval mode: identity
            self.mask = torch.ones_like(X, device=self.device)
            return X

    def backward(self, dY):
        if self.training and self.p > 0.0:
            # Only pass gradients where mask kept activations
            return dY * self.mask
        else:
            # Eval mode: gradient flows unchanged
            return dY

    def update(self, lr):
        # No learnable params in Dropout
        pass




class CrossEntropyFromLogits:
    """
    Implements the combination of:
    - Softmax activation (from raw logits)
    - Cross-entropy loss

    This is a common choice for multi-class classification.
    """

    def forward(self, Z, Y):
        """
        Forward pass: compute the cross-entropy loss from raw logits.

        Args:
            Z (torch.Tensor): Logits (unnormalized scores) of shape (batch_size, n_classes).
            Y (torch.Tensor): True class indices of shape (batch_size,).

        Returns:
            loss torch.Tensor: Scalar value of the cross-entropy loss.
        """
        self.Y = Y  # Store true labels for backward pass

        # TODO: Compute softmax probabilities (convert logits to probabilities)
        self.A = torch.softmax(Z, dim=1)

        # TODO: Compute log-softmax (log probabilities)
        log_softmax_Z = torch.log_softmax(Z, dim=1)

        # TODO: Select the log-probabilities of the correct classes for each sample
        log_probs = log_softmax_Z[torch.arange(Z.shape[0]), Y]

        # TODO: Cross-entropy loss: average negative log-likelihood
        loss = -log_probs.mean()
        
        return loss

    def backward(self, n_classes):
        """
        Backward pass: compute the gradient of the loss with respect to logits Z.

        Args:
            n_classes (int): Number of classes in the classification problem.

        Returns:
            torch.Tensor: Gradient dZ of shape (batch_size, n_classes).
        """

        m = self.Y.shape[0]

        # TODO: One-hot encode the true labels
        Y_one_hot = torch.nn.functional.one_hot(self.Y, num_classes=n_classes).float()

        # TODO: Derivative of cross-entropy w.r.t logits: softmax_output - one_hot_labels
        dZ = (self.A - Y_one_hot) / m

        return dZ