# �v���H�y�E���t��

�o�Өt�Υi�H�q�v�����^���H�y�A�ϥ�FaceNet�p��S�x�V�q�A�M��q�LChinese Whispers��k�i��E���A�H��X�v�����ۦP�H�������P�H�y�Ϲ��C

## �\�෧�z

1. **�H�y�^��**�G�ϥ�MTCNN�q�v�����^���H�y
2. **�S�x����**�G�ϥ�FaceNet�N�H�y�ഫ���S�x�V�q
3. **�H�y�E��**�G�ϥ�Chinese Whispers��k�i��E��
4. **���߭p��**�G�p��C�ӻE�������ߧ@���H�y�w
5. **���G�i����**�G�ЫئU�إi���ƨӮi�ܻE�����G

## �t�λݨD

- Python 3.7+
- TensorFlow 1.x (�ѩ�ϥΪ��OFaceNet��l�ҫ��A�ݭn�ϥ�TensorFlow 1.x)
- OpenCV
- NetworkX
- Matplotlib
- NumPy
- tqdm

## �w�˫��n

1. �J���s�x�w�G
   ```bash
   git clone https://github.com/yourusername/video-face-clustering.git
   cd video-face-clustering
   ```

2. �w�˨̿�G
   ```bash
   pip install -r requirements.txt
   ```

3. �U��FaceNet�w�V�m�ҫ��G
   ```bash
   # �i�H�qGitHub��FaceNet�s�x�w�U��
   # https://github.com/davidsandberg/facenet
   ```

## �ϥΤ�k

### �򥻥Ϊk

```bash
python main.py --input_video �v�����|.mp4 --output_dir ��X�ؿ� --model_dir FaceNet�ҫ��ؿ�
```

### ����Ѽ�

- `--input_video`�G��J�v�����|
- `--output_dir`�G��X�ؿ�
- `--model_dir`�GFaceNet�ҫ��ؿ�
- `--batch_size`�G��q�j�p�]�q�{100�^
- `--face_size`�G�H�y�Ϲ��j�p�]�q�{160�^
- `--cluster_threshold`�G�E���H�ȡ]�q�{0.7�^
- `--frames_interval`�G�^���V�����j�]�q�{30�^
- `--visualize`�G�O�_�i���Ƶ��G�]�q�{�_�^

## ��X�ؿ����c

�B�z������A��X�ؿ��N�]�t�H�U���e�G

```
��X�ؿ�/
�u�w�w faces/                # �q�v�����^�����Ҧ��H�y
�u�w�w clusters/             # �E�����G
�x   �u�w�w cluster_0/        # �E��0���H�y
�x   �u�w�w cluster_1/        # �E��1���H�y
�x   �|�w�w ...
�u�w�w centers/              # �E�����߫H��
�x   �|�w�w centers_data.pkl  # ���߼ƾڡ]�]�A�s�X�^
�|�w�w visualization/        # �i���Ƶ��G
    �u�w�w cluster_sizes.png           # �E���j�p���G��
    �u�w�w cluster_0_network.png       # �E��0��������
    �u�w�w cluster_0_thumbnails.png    # �E��0���Y���϶�
    �u�w�w ...
    �|�w�w cluster_centers.png         # �Ҧ��E������
```

## �Ҳջ���

- `main.py`�G�D�{�ǡA��X�Ҧ��B�J
- `face_detection.py`�G�H�y�˴��Ҳ�
- `feature_extraction.py`�G�S�x�����Ҳ�
- `clustering.py`�G�E���Ҳ�
- `visualization.py`�G�i���ƼҲ�

## �d��

```bash
python main.py --input_video �q�v.mp4 --output_dir ./results --model_dir ./models/facenet --visualize
```

## �޳N�Ӹ`

- **MTCNN**�G�h���ȯ��p���n�����A�Ω�H�y�˴��M���
- **FaceNet**�G�Ω�N�H�y�ഫ���S�x�V�q�]�O�J�^
- **Chinese Whispers**�G�@�ذ��Ϫ��E����k�A�S�O�A�Ω�H�y�E��
- **�E�����߭p��**�G�i�H�ϥΥ����ȩγ̤p�����Z���ӭp�⤤��

## �G�ٱư�

- �p�G�X�{���s���~�A�i�H��֧�q�j�p�]`--batch_size`�^
- �p�G�E�����G���z�Q�A�i�H�վ�E���H�ȡ]`--cluster_threshold`�^
- �p�G�B�z�t�פӺC�A�i�H�W�[�V���j�]`--frames_interval`�^

## �����i��V

1. �K�[�H�y�ѧO�\��A�N�E�����G�P�w���H���ƾڮw�ǰt
2. ��i�E����k�A�������
3. �u�ƳB�z�t�סA�����ɳB�z
4. �K�[GUI�ɭ�
5. ����h�Ӽv������q�B�z
