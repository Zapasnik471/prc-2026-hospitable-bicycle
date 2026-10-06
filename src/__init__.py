import warnings

# polars cannot verify sortedness of as-of joins with `by` groups; all inputs are sorted explicitly.
warnings.filterwarnings("ignore", message="Sortedness of columns cannot be checked")
