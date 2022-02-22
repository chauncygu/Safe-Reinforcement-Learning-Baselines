#### Safety Multi-agent Mujoco 


## 1. Sate Many Agent Ant

According to Zanger's work, 

The reward function is equal to the rewards in the common Ant-v2 environment and comprises the torso velocity in global x-direction, a negative control reward on exerted torque, a negative contact reward and a constant positive reward for survival, which results in

<img src="https://latex.codecogs.com/png.image?\dpi{110}&space;r=\frac{x_{\text&space;{torso&space;},&space;t&plus;1}-x_{\text&space;{torso&space;},&space;t}}{d&space;t}-\frac{1}{2}\left\|\boldsymbol{a}_{t}\right\|_{2}^{2}-\frac{1}{2&space;*&space;10^{3}}&space;\|&space;\text&space;{&space;contact&space;}_{t}&space;\|_{2}^{2}&plus;1" title="r=\frac{x_{\text {torso }, t+1}-x_{\text {torso }, t}}{d t}-\frac{1}{2}\left\|\boldsymbol{a}_{t}\right\|_{2}^{2}-\frac{1}{2 * 10^{3}} \| \text { contact }_{t} \|_{2}^{2}+1" />

```python
xposafter = self.get_body_com("torso_0")[0]
forward_reward = (xposafter - xposbefore)/self.dt
ctrl_cost = .5 * np.square(a).sum()
contact_cost = 0.5 * 1e-3 * np.sum(np.square(np.clip(self.sim.data.cfrc_ext, -1, 1)))
survive_reward = 1.0
        
reward = forward_reward - ctrl_cost - contact_cost + survive_reward
```

And the cost,


<img src="https://latex.codecogs.com/png.image?\dpi{110}&space;c=&space;\begin{cases}0,&space;&&space;\text&space;{&space;for&space;}&space;\quad&space;0.2&space;\leq&space;z_{\text&space;{torso&space;},&space;t&plus;1}&space;\leq&space;1.0&space;\\&space;&&space;\text&space;{&space;and&space;}\left\|\boldsymbol{x}_{\text&space;{torso&space;},&space;t&plus;1}-\boldsymbol{x}_{\text&space;{wall&space;}}\right\|_{2}&space;\geq&space;1.8&space;\\&space;1,&space;&&space;\text&space;{&space;else&space;}\end{cases}" title="c= \begin{cases}0, & \text { for } \quad 0.2 \leq z_{\text {torso }, t+1} \leq 1.0 \\ & \text { and }\left\|\boldsymbol{x}_{\text {torso }, t+1}-\boldsymbol{x}_{\text {wall }}\right\|_{2} \geq 1.8 \\ 1, & \text { else }\end{cases}" />



```python
yposafter = self.get_body_com("torso_0")[1]
ywall = np.array([-5, 5])
if xposafter < 20:
  y_walldist = yposafter - xposafter * np.tan(30 / 360 * 2 * np.pi) + ywall
elif xposafter>20 and xposafter<60:
  y_walldist = yposafter + (xposafter-40)*np.tan(30/360*2*np.pi) - ywall
elif xposafter>60 and xposafter<100:
  y_walldist = yposafter - (xposafter-80)*np.tan(30/360*2*np.pi) + ywall
else:
  y_walldist = yposafter - 20*np.tan(30/360*2*np.pi) + ywall
obj_cost = (abs(y_walldist) < 1.8).any() * 1.0

body_quat = self.data.get_body_xquat('torso_0')
 z_rot = 1-2*(body_quat[1]**2+body_quat[2]**2)  ### normally xx-rotation, not sure what axes mujoco uses

state = self.state_vector()
notdone = np.isfinite(state).all() \
                      and state[2] >= 0.2 and state[2] <= 1.0\
                      and z_rot>=-0.7 #ADDED
done = not notdone
done_cost = done * 1.0

cost = np.clip(obj_cost + done_cost, 0, 1)
```


[1] Zanger, Moritz A., Karam Daaboul, and J. Marius Zöllner. 2021. “Safe Continuous Control with Constrained Model-Based Policy Optimization.” arXiv [cs.LG]. arXiv. http://arxiv.org/abs/2104.06922.
