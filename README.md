# Skin cancer detection with deep neural network

## CorgiVision group project

### Participants:

* Barbara Kéri - AR5KHR
* Benjámin Csontó - JTB4Y1
* Sámuel Csányi - I7ULKV

### Kaggle ISIC 2024 challange 

In this project, we will develop image-based algorithms to identify histologically confirmed skin cancer cases with single-lesion crops from 3D total body photos (TBP). The image quality resembles close-up smartphone photos, which are regularly submitted for telehealth purposes. We have a large dataset of images, and for every image, we can use some metadata about the patient and the leisure.

### Used metadata:

| Name                        | Description                                                  | Feature scaling technique |
| --------------------------- | ------------------------------------------------------------ | ------------------------- |
| age_approx                  | Approximate age of patient at time of imaging.               | Z-score scaling           |
| sex                         | Sex of the person.                                           | -                         |
| tbp_lv_areaMM2              | Area of lesion (mm^2).+                                      | Min-Max Scaling           |
| tbp_lv_area_perim_ratio     | Border jaggedness, the ratio between lesions perimeter and area. Circular lesions will have low values; irregular shaped lesions will have higher values. Values range 0-10.+ | Min-Max Scaling           |
| tbp_lv_color_std_mean       | Color irregularity, calculated as the variance of colors within the lesion's boundary. | Min-Max Scaling           |
| tbp_lv_deltaLBnorm          | Contrast between the lesion and its immediate surrounding skin. Low contrast lesions tend to be faintly visible such as freckles; high contrast lesions tend to be those with darker pigment. Calculated as the average delta L*B* of the lesion relative to its immediate background in L*A*B* color space. Typical values range from 5.5 to 25.+ | Min-Max Scaling           |
| tbp_lv_location             | Classification of anatomical location, divides arms & legs to upper & lower; torso into thirds.+ | One-Hot encoding          |
| tbp_lv_minorAxisMM          | Smallest lesion diameter (mm).+                              | Min-Max Scaling           |
| tbp_lv_nevi_confidence      | Nevus confidence score (0-100 scale) is a convolutional neural network classifier estimated probability that the lesion is a nevus. The neural network was trained on approximately 57,000 lesions that were classified and labeled by a dermatologist.+,++ | Min-Max Scaling           |
| tbp_lv_norm_border          | Border irregularity (0-10 scale); the normalized average of border jaggedness and asymmetry.+ | Min-Max Scaling           |
| tbp_lv_norm_color           | Color variation (0-10 scale); the normalized average of color asymmetry and color irregularity.+ | Min-Max Scaling           |
| tbp_lv_perimeterMM          | Perimeter of lesion (mm).+                                   | Min-Max Scaling           |
| tbp_lv_radial_color_std_max | Color asymmetry, a measure of asymmetry of the spatial distribution of color within the lesion. This score is calculated by looking at the average standard deviation in L*A*B* color space within concentric rings originating from the lesion center. Values range 0-10.+ | Min-Max Scaling           |
| tbp_lv_symm_2axis           | Border asymmetry; a measure of asymmetry of the lesion's contour about an axis perpendicular to the lesion's most symmetric axis. Lesions with two axes of symmetry will therefore have low scores (more symmetric), while lesions with only one or zero axes of symmetry will have higher scores (less symmetric). This score is calculated by comparing opposite halves of the lesion contour over many degrees of rotation. The angle where the halves are most similar identifies the principal axis of symmetry, while the second axis of symmetry is perpendicular to the principal axis. Border asymmetry is reported as the asymmetry value about this second axis. Values range 0-10.+ | Min-Max Scaling           |
| tbp_lv_symm_2axis_angle     | Lesion border asymmetry angle.+                              | Min-Max Scaling           |

Files in the repo: 

* **explore.ipynb:** The main project file
* **compute_server_connection:** Instructions how to connect to our GCP Compute Engine  

