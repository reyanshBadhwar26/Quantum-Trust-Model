import textstat
import numpy as np

def readability_score(text):
    """Computes a single readability score in the range [-1,1] using a simple weighted average"""
    
    # Get readability metrics
    flesch = textstat.flesch_reading_ease(text)  # Higher = easier
    fk_grade = textstat.flesch_kincaid_grade(text)  # Lower = easier
    gunning_fog = textstat.gunning_fog(text)  # Lower = easier
    smog = textstat.smog_index(text)  # Lower = easier
    ari = textstat.automated_readability_index(text)  # Lower = easier
    dale_chall = textstat.dale_chall_readability_score(text)  # Lower = easier

    # Convert all metrics to the same scale (higher = easier)
    max_flesch = 100
    max_grade = 20  # Approximate max grade level
    max_dale_chall = 10

    normalized_scores = [
        flesch / max_flesch,  # Already 0-1
        1 - (fk_grade / max_grade),
        1 - (gunning_fog / max_grade),
        1 - (smog / max_grade),
        1 - (ari / max_grade),
        1 - (dale_chall / max_dale_chall),
    ]

    print(normalized_scores)

    # Ensure values are within [0,1]
    normalized_scores = [max(min(s, 1), 0) for s in normalized_scores]

    combined_score = np.mean(normalized_scores)

    # Scale to [-1,1]
    return 2 * combined_score - 1
