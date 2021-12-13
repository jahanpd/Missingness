# Dealing with missing data through attention and latent space regularization

A principled strategy for training and performing inference on observed data only, without imputation or dropping rows. 

## Experiments
We tested the properties of the latent space model using a synthetic dataset and benchmarked the method on real world datasets.

### Spiral Dataset Experiment 1
Showing the properties of the latent space using the binary classification synthetic spiral dataset augmented with 2 additional variables.
These 2 variables are a purely noise variable containing no information about the outcome, and a signal variable with some information.
We show that the noise variable is represented as very close to the point of maximal uncertainty in the latent space, whereas the signal variable is further away in 'informative' space.

### Spiral Dataset Experiment 2
A further exploration of the latent space and feature important representation in the setting of missingness.
We show that the distance in the latent space between an informative variable and uninformative variable decreases as missingness increases in the informative variable.
Additionally, we show that the feature importance as defined in the concrete dropout layer decreases as missingness increases.

### OpenML Benchmarking
We benchmark the performance of this approach using OpenML Benchmarking datasets with complete datasets and the same datasets that are corrupted with three types of missingness pattern (MCAR, MAR and MNAR).
As a comparison, we use the popular and highly performing Light-GBM that handles missingness out-of-the-box.
We also compare the out-of-the-box missingness handling of these algorithms against an impute and regress strategy.
Imputation strategies tested include simple imputation, multivariate imputation, and multiple imputation with random forests.

## Requirements
Please install a version of [Jax](https://github.com/google/jax) appropriate for your system (eg. GPU enabled), then install the requirements in the requirements.txt file.

## BibTeX

    @InProceedings{penny-dimri2021missing,
        author = {Jahan Penny-Dimri and Christoph Bergmeir etc},
        title = {Dealing with missing data through attention and latent space regularization},
        year = {2021}
    }

## :e-mail: Contact
If you have any questions, please email `jahan.penny-dimri@monash.edu`.
