import numpy as np


def get_alpha_bar(betas: np.ndarray) -> np.ndarray:
    """
    Compute cumulative product of (1 - beta).

    Args:
        betas: Array of beta values for each timestep

    Returns:
        alpha_bar: Cumulative product array
    """
    alphas = 1.0 - betas
    alpha_bar = np.cumprod(alphas)
    return alpha_bar


def forward_diffusion(
    x_0: np.ndarray,
    t: int,
    betas: np.ndarray
) -> tuple:
    """
    Sample x_t from q(x_t | x_0)

    Args:
        x_0: Original image/data (numpy array)
        t: timestep (int)
        betas: noise schedule (numpy array)

    Returns:
        x_t: Noisy version at timestep t
        noise: Noise added
    """
    # Get alpha_bar values
    alpha_bar = get_alpha_bar(betas)

    # Sample random noise
    noise = np.random.randn(*x_0.shape)

    # Compute x_t using reparameterization trick
    x_t = (
        np.sqrt(alpha_bar[t]) * x_0 +
        np.sqrt(1.0 - alpha_bar[t]) * noise
    )

    return x_t, noise


# -------------------------------
# Example Usage
# -------------------------------
if __name__ == "__main__":
    # Create dummy "image" (e.g., 4x4)
    x_0 = np.random.randn(4, 4)

    # Define timesteps
    T = 10

    # Create beta schedule (small increasing noise)
    betas = np.linspace(0.0001, 0.02, T)

    # Choose timestep
    t = 5

    x_t, noise = forward_diffusion(x_0, t, betas)

    print("Original x_0:\n", x_0)
    print("\nNoisy x_t at timestep", t, ":\n", x_t)
    print("\nNoise added:\n", noise)