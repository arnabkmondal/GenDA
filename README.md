# GenDA

1. Download FFHQ checkpoint using the following link (This is a third party link (owner: Kim Seonghyeon) and does not reveal author identity):
https://drive.google.com/file/d/1PQutd-JboOCOZqmd95XWxWrO8gGEvRcO/view

2. Place the checkpoint in the folder checkpoint.

3. Download the 10 shot babies dataset using the following link (This is a third party link (owner: Utkarsh Ojha) and does not reveal author identity):
https://drive.google.com/file/d/1JygAunIzpMyRA9kPXobWSuwzVw7oOHU6/view?usp=sharing

4. Download the entire babies dataset for FID computation using the following link (This is a third party link (owner: Utkarsh Ojha) and does not reveal author identity):
https://drive.google.com/file/d/1JmjKBq_wylJmpCQ2OWNMy211NFJhHHID/view?usp=sharing

5. Create an environment as per the specifications of requirements.yml

6. Activate the newly created environment.

7. Specify the path of 10 shot babies images in the line 7 of preprocess_babies_data.py file and run it to create babies_training.npy

8. Change directory to few_shot_expts

9. Run NLI_msmt.sh

10. Compute FID using pytorch-fid (https://pypi.org/project/pytorch-fid/)
