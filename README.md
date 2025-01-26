# t-SNE Python Implementation

This repository contains my implementation of the **t-Distributed Stochastic Neighbor Embedding (t-SNE)** algorithm, completed as part of the **Advanced Machine Learning** course at the **Armenian Code Academy**. This project highlights my skills in Machine Learning, Python programming, and hands-on implementation of complex algorithms.


### What is t-SNE?

t-SNE is a powerful dimensionality reduction technique commonly used for visualizing high-dimensional datasets in a lower-dimensional space, typically 2D or 3D. The algorithm preserves local relationships between data points, making it especially effective for clustering and visualizing patterns in datasets. 

Some key applications of t-SNE include:
- Visualizing the structure of high-dimensional data.
- Understanding clustering patterns in datasets.
- Preprocessing data for downstream Machine Learning tasks.


### Features of This Implementation

1. **Custom Implementation**: The t-SNE algorithm was implemented from scratch, without relying on external libraries like `sklearn` or `openTSNE`.
2. **Optimized High-Dimensional Affinities**: Pairwise affinities in the original space are calculated using Gaussian kernels with adaptive bandwidth, ensuring an accurate representation of local structures.
3. **Gradient Descent**: Low-dimensional embeddings are optimized using a custom gradient descent algorithm.
4. **MNIST Visualization**: The implementation is tested on the MNIST dataset, showcasing the separation of handwritten digit classes in 2D space.
5. **Step-by-Step Processing**:
   - Initialization of low-dimensional points.
   - Calculation of symmetric affinities in high and low-dimensional spaces.
   - Iterative optimization with cost monitoring.


### Project Outcomes

The implementation was tested on the MNIST dataset, where 1,000 handwritten digit samples were reduced from 784 dimensions to a 2D representation. The results clearly visualize distinct clusters corresponding to each digit, demonstrating the power of t-SNE in understanding high-dimensional data.
