# CSI-IMD Dataset

CSI-IMD (Characterizing Semantically Impactful Image Manipulations Dataset) is a benchmark dataset introduced for studying image manipulation detection and localization under varying levels of semantic significance. Unlike traditional forensic datasets that primarily focus on pixel-level alterations, CSI-IMD provides additional semantic annotations that characterize the importance and impact of manipulations. The dataset contains a manually curated Gold Standard set comprising 1,000 pristine and 1,000 manipulated images with six annotation types, together with a large-scale expanded set containing approximately 500,000 manipulated images generated through an automated pipeline. Each manipulated image is associated with pixel-level masks and semantic relevance information, enabling both image-level detection and pixel-level localization studies. The dataset is particularly designed to analyze how semantic significance influences the performance of forensic models.

## Resources

* **Paper (WACV 2025)**
  A Semantically Impactful Image Manipulation Dataset: Characterizing Image Manipulations using Semantic Significance [[Paper](https://openaccess.thecvf.com/content/WACV2025/papers/Chen_A_Semantically_Impactful_Image_Manipulation_Dataset_Characterizing_Image_Manipulations_using_WACV_2025_paper.pdf)]

* **Dataset Repository**
  GitHub repository containing the Gold Standard and Expanded CSI-IMD datasets. [[Link](https://github.com/csiimd/csiimd)]

## Dataset Structure

* `manipulated_images/`
  Manipulated images corresponding to semantically meaningful edits.

* `pristine_images_used_for_manipulation/`
  Original pristine images from which manipulated samples were generated.

* `pixel_based_truth_mask/`
  Pixel-level ground-truth masks for localization evaluation.

* `semantic_relevant_pixel_truth_mask/`
  Masks highlighting semantically relevant manipulated regions.

* `image_info.csv`
  Metadata describing image relationships and semantic annotations.

## Statistics

* Gold Standard Set

  * 1,000 pristine images
  * 1,000 manipulated images
  * Six annotation types

* Expanded Set

  * Approximately 500,000 manipulated images

