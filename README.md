# Student Behavioral Segmentation using DBSCAN

An unsupervised machine learning project using Python to discover hidden behavioral patterns among 300 university students through density-based clustering and dimensionality reduction.

## Project Description
Developed as an individual final project for a Data Clustering course, this project explored how data-driven approaches can uncover hidden patterns in student behavior. Traditional educational approaches often treat students as a single group, even though each student has different habits, support systems, and learning behaviors. 

Using the Density-Based Spatial Clustering of Applications of Noise (DBSCAN) algorithm, I analyzed survey data from 300 university students to identify distinct behavioral clusters and generate meaningful insights. This experience demonstrated that every individual has unique characteristics that cannot always be understood through a single perspective, while also highlighting the importance of building solutions that are more adaptive, thoughtful, and centered on real human behavior.

## Tech Stack & Advanced Workflow
- **Programming Language:** Python
- **Data Manipulation:** `pandas`, `numpy`
- **Data Preprocessing:** Missing value handling & feature scaling via `StandardScaler`
- **Dimensionality Reduction:** Principal Component Analysis (PCA) for 2D visualization
- **Machine Learning Model:** DBSCAN (Density-Based Spatial Clustering of Applications of Noise)
- **Evaluation Metrics:** Silhouette Score & K-Distance Optimization

##  Step-by-Step Data Engineering Pipeline

### 1. Data Cleaning & Feature Scaling
The raw survey dataset was checked for completeness. Because clustering algorithms calculate density based on spatial distance, all behavioral features were standardized using `StandardScaler` to bring them onto a uniform scale (Mean = 0, Variance = 1).

### 2. Dimensionality Reduction (PCA)
To visualize the multi-dimensional behavioral features, Principal Component Analysis (PCA) was applied to compress the variables into two main components (`PC1` and `PC2`) while retaining maximum data variance.

### 3. DBSCAN Clustering & Hyperparameter Tuning
Instead of using K-Means (which forces every point into a cluster), DBSCAN was deployed. The optimal hyperparameter for **Epsilon ($\epsilon$)** was mathematically determined using a **K-Distance Plot**. By locating the maximum curvature (the "elbow" point), the threshold was set at **1.20** to effectively separate core clusters from background noise.

## Evaluation & Visual Insights

### 1. K-Distance Plot (Hyperparameter Optimization)
The sorted K-Distance plot below demonstrates the threshold selection for Epsilon ($\epsilon = 1.20$). Data points resting within the shaded pink region represent the isolated anomalous behaviors classified as background noise:

<img width="817" height="310" alt="1" src="https://github.com/user-attachments/assets/fba7ad2d-199e-4c9d-b76a-22e5d468d140" />

### 2. PCA 2D Scatter Plot & Cluster Distribution
Following the PCA transformation, a 2D scatter plot was generated to map out the spatial distribution of the student segments. The algorithm successfully identified core densities while isolating scattered outlier behaviors:

- **Cluster -1 (Noise/Outlier):** 195 students (represented in grey). This highlights the high behavioral diversity in the survey data that does not fit into standard categories.
- **Cluster 0:** 55 students (red)
- **Cluster 1:** 23 students (blue)
- **Cluster 2:** 23 students (green)
- **Cluster 3:** 4 students (orange)

<img width="726" height="551" alt="2" src="https://github.com/user-attachments/assets/7af99cb2-dbb4-4637-8a7b-a1516d8694ca" />

### 3. Quantitative Cluster Distribution
The bar chart below illustrates the exact quantitative distribution of university students across the identified DBSCAN clusters, highlighting the significant volume of outlier data points isolated by the density model:

<img width="731" height="317" alt="3" src="https://github.com/user-attachments/assets/67a2c790-773c-4391-957d-014fb97c5926" />

### 4. Cluster Characteristic Heatmap (Behavioral Profiling)
To interpret the real-world meaning of each group, a heatmap representing the mean feature values per DBSCAN cluster was generated:

- **Cluster 0:** Students characterized by high family support (~4.82) and low social media usage (~0.93).
- **Cluster 1:** Active social media users (~2.83) with moderate exam approaches.
- **Cluster 2:** Students with a higher financial status index (~2.00) and structured exam preparation habits.
- **Cluster 3:** A niche group (4 students) showcasing maximum reliance on lecturer mentorship (~4.00) paired with solid social media engagement (~3.00).

<img width="663" height="408" alt="4" src="https://github.com/user-attachments/assets/83ece664-caad-4e33-b829-84d3d13713b7" />

### 5. Silhouette Coefficient
The model's cluster cohesion and boundary separation were mathematically verified using the Silhouette Score via `silhouette_score(X_scaled, labels)`.

## Repository Structure
- `data_univstudent_300_final.csv` - The raw demographic and behavioral survey dataset.
- `student_clustering.py` - The complete Python pipeline script documenting preprocessing, PCA, DBSCAN model configuration, and evaluation.
