#### SoftREPA trained on paired COCO dataset (118K imgs)
#### Deepfashion dataset (25K imgs)
![alt text](Architecture.png)
![alt text](Scoring_Module.png)


#### SoftREPA also used Diffusion-DPO(Direct Preference Optimization for Diffusion), DDPO (Denoising Diffusion Policy Optimization)
![alt text](image.png)

$$\mathcal{L}_{DPO}(\theta; \theta_{\text{ref}}) = -\mathbb{E}_{(x_w, x_l, c)} \left[ \log \sigma \left( \beta \cdot (\text{err}(x_l, \theta) - \text{err}(x_l, \theta_{\text{ref}})) - \beta \cdot (\text{err}(x_w, \theta) - \text{err}(x_w, \theta_{\text{ref}})) \right) \right]$$
![alt text](image-1.png)

![alt text](image-2.png)


##### SoftREPA Reward function
- Reward model selection
![alt text](image-3.png)
- Reward score
![alt text](image-4.png)
- mean(Reward score)
![alt text](image-5.png)


#####  angle condition
- yaw > 40 | yaw < -40 (turn his/her head to his/her left/right over the shoulder)
- yaw > 20 | yaw < -20 (turn his/her head to his/her left/right)
- yaw < 20 & yaw > -20 (face forward)
- pitch > 10 (look up)
- pitch < -10 (look down)


## 預計測量比較:
- 設定15種posture prompt，比較不同T2I model(PhotoMaker v2, UniProtrait, PuLID等)
- 評估指標: CLIP, DINO, HPS, ImageReward, FID(SoftREPA用COCO-val 1K), LPIPS, Latency 

