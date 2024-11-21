ORB feature extracter, Brute force matcher를 사용하여, 52개의 image data에서 Essential matrix를 얻은 다음 R, t matrix로 분해하여 visual odometry를 하는 작업을 수행하였습니다.

### gray scale image
!["gray scale image"](https://github.com/kimmyeongjune/Autonomous-Driving-Computing-HW4/blob/main/gray_1.png)

### rgb image
!["rgb image"](https://github.com/kimmyeongjune/Autonomous-Driving-Computing-HW4/blob/main/rgb_1.png)

### ORB detector를 사용한 key points feature extract
!["ORB detector를 사용한 key points feature extract"](https://github.com/kimmyeongjune/Autonomous-Driving-Computing-HW4/blob/main/key_points_1.png)

### BFmatcher를 사용한 kp matching 
!["BFmatcher를 사용한 kp matching "](https://github.com/kimmyeongjune/Autonomous-Driving-Computing-HW4/blob/main/feature_matched_frame0_1.png)

### estimate rot@translation matrix
!["estimate rot@translation matrix"](https://github.com/kimmyeongjune/Autonomous-Driving-Computing-HW4/blob/main/estimated_rotation&translation.png)

### estimate_camera motion
!["estimate_camera motion"](https://github.com/kimmyeongjune/Autonomous-Driving-Computing-HW4/blob/main/estimate_motion_1.png)
!["estimate_camera motion"](https://github.com/kimmyeongjune/Autonomous-Driving-Computing-HW4/blob/main/estimate_motion_2.png)

### Trajectory matrix
!["Trajectory matrix"](https://github.com/kimmyeongjune/Autonomous-Driving-Computing-HW4/blob/main/Trajectory_matrix.png)

### visual odometry
!["visual odometry"](https://github.com/kimmyeongjune/Autonomous-Driving-Computing-HW4/blob/main/visual_odometry.png)

### visual odometry matrix
!["visual odometry matrix"](https://github.com/kimmyeongjune/Autonomous-Driving-Computing-HW4/blob/main/visual_odometry_matrix.png)

### visual odometry 영상
!["visual odometry"](https://github.com/kimmyeongjune/Autonomous-Driving-Computing-HW4/blob/main/visual_odometry.gif)

### 다양한 descriptor를 사용하여 feature extract 및 시간, kp 측정

### ORB
!["ORB"](https://github.com/kimmyeongjune/Autonomous-Driving-Computing-HW4/blob/main/ORB_1.png)

### SIFT
!["SIFT"](https://github.com/kimmyeongjune/Autonomous-Driving-Computing-HW4/blob/main/SIFT_1.png)

### BRISK
!["BRISK"](https://github.com/kimmyeongjune/Autonomous-Driving-Computing-HW4/blob/main/BRISK_1.png)

### Ablation
!["Ablation"](https://github.com/kimmyeongjune/Autonomous-Driving-Computing-HW4/blob/main/feature_descriptor_ablation.png)
