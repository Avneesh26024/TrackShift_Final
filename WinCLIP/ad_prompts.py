state_level_normal_prompts = [
    '{}',
    'flawless {}',
    'perfect {}',
    'unblemished {}',
    '{} without flaw',
    '{} without defect',
    '{} without damage'
 #'three flawless, perfect and unblemished {} with different colors without any defect, flaw or damage in a bowl',
#    'flawless, perfect and unblemished {} without any defect, flaw or damage'
]

state_level_abnormal_prompts = [
 #   'damaged {} with flaw or defect or damage',
    'damaged {}',
    '{} with flaw',
    '{} with defect',
    '{} with damage',
 ##   '{} with missing parts'
 ##   '{} with print',  # added
 ##    '{} with hole',  # added
 ##   '{} with crack', # added
 ##   '{} with scratch', # added
 ##    '{} with discoloration',
 ##    '{} with stains',
 ##    '{} with missing parts',
 ##    '{} with broken parts',
 ##    '{} with bumpy surfaces'
]

#DIY_list = [
#    'cable'
#    ]

#state_level_normal_prompts = [
# 'three flawless, perfect and unblemished {} with different colors without any defect, flaw or damage in a bowl'
# 'a flawless, perfect and unblemished {} without any defect, flaw or damage'
#]
#'an electronic device sitting on top of a wooden board'
#state_level_abnormal_prompts = [
#    'damaged {} with flaw or defect or damage'
#]

state_level_normality_specific_prompts = [
    # needed to be designed category by category
]

state_level_abnormality_specific_prompts = [
    '{} with {} defect', # object, anomaly type
    '{} with {} flaw',
    '{} with {} damage',
]

template_level_prompts = [
    'a cropped photo of the {}',
    'a cropped photo of a {}',
    'a close-up photo of a {}',
    'a close-up photo of the {}',
    'a bright photo of a {}',
    'a bright photo of the {}',
    'a dark photo of the {}',
    'a dark photo of a {}',
    'a jpeg corrupted photo of a {}',
    'a jpeg corrupted photo of the {}',
    'a blurry photo of the {}',
    'a blurry photo of a {}',
    'a photo of a {}',
    'a photo of the {}',
    'a photo of a small {}',
    'a photo of the small {}',
    'a photo of a large {}',
    'a photo of the large {}',
    'a photo of the {} for visual inspection',
    'a photo of a {} for visual inspection',
    'a photo of the {} for anomaly detection',
    'a photo of a {} for anomaly detection'
]


# === F1 / Motorsports specific prompt extensions ===
# These are additional prompts tailored for racing circuits, tyres, curbs, barriers and
# other Formula-1 related objects. They are meant to be used alongside the generic
# state-level and template-level prompts above by substituting the object/class name
# (e.g. 'race track', 'tyre', 'curb', 'wheel rim', 'run-off area').

f1_state_level_normal_prompts = [
    '{}',
    'clean {}',
    'dry {}',
    'dry racing {}',
    'well-maintained {}',
    'clear racing line on {}',
    '{} without debris',
    '{} without oil or fluid',
    'no tyre marbles on {}',
    'pit-lane {} in good condition'
]

f1_state_level_abnormal_prompts = [
    '{} with skid marks',
    '{} with heavy skid marks',
    '{} with rubber deposits',
    '{} with tyre marbles',
    '{} with oil spill',
    '{} with fluid leak',
    '{} with debris on track',
    '{} with standing water',
    '{} with a puncture',
    '{} with rim damage',
    '{} with cracked tarmac',
    '{} with pothole',
    '{} with curb damage',
    '{} with barrier damage',
    '{} with paint transfer or scrape',
    '{} with scuffing or abrasion',
    '{} with melted rubber',
    '{} with surface glazing or polishing',
    '{} with foreign object debris'
]

f1_state_level_abnormality_specific_prompts = [
    '{} with {} on the apex',
    '{} with {} on the racing line',
    '{} with {} near the curb',
    '{} with {} on the left side',
    '{} with {} on the right side',
    'close-up of {} showing {}',
    'patch of {} containing {}',
    'section of {} with visible {}',
    '{} at the pit entry with {}',
    '{} at the finish line with {}'
]

# Add a few F1-specific template prompts to help contextualize camera views
f1_template_level_prompts = [
    'a drone view of the {}',
    'a broadcast camera view of the {}',
    'a pit-lane close-up of the {}',
    'a close-up photo of the curb of the {}',
    'a high-resolution photo of the {}',
    'a telephoto shot of the {}',
    'an onboard camera view of the {}'
]

# Merge suggestions: users of the model can combine the F1 lists with the generic
# lists above when building text galleries for classes like 'race track', 'tire',
# 'curb', 'barrier', 'run-off area', 'wheel rim' etc. Example usage in code:
# prompts = state_level_abnormal_prompts + f1_state_level_abnormal_prompts