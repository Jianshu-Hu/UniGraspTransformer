"""
MujocoSimEnv Version 0.2
2025.04.06
"""

import mujoco
import numpy as np
import mujoco.viewer
import time

class MujocoSimEnv:
    def __init__(self, model_path):
        """
        MujocoSimEnv Version 0.2
        2025.04.06

        Methods:
        reset(self) 重置仿真环境
        step(self) 进行一步仿真
        render(self) 渲染当前帧
        quit(self) 退出仿真环境
        position_ctrl(self, actuator_id, target_pos) 施加位置驱动器控制(需要XML模型中的position actuator)
        position_set(self, joint_id, new_pos) 直接设置当前位置

        Variables:
        self.joint_data 关节信息列表
        self.step_count 步数统计
        """

        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.data = mujoco.MjData(self.model)
        self.joint_pos = self.get_joint_pos()
        self.joint_vec = self.get_joint_vec()
        self.viewer = None
        self.step_count = 0

    def reset(self):
        """ 重置环境 """
        mujoco.mj_resetData(self.model, self.data)
        self.step_count = 0

    def step(self):
        """ 执行一步仿真 """
        mujoco.mj_step(self.model, self.data)
        self.joint_pos = self.get_joint_pos()
        self.joint_vec = self.get_joint_vec()
        self.step_count += 1
        # print(f"Step: {self.step_count}")

    def get_joint_pos(self):
        """ 
        获取当前关节数据
        [(name, angle),...]
        """

        # 获取关节状态数据
        joint_positions = self.data.qpos.copy()

        # 构建 joint_data 列表
        joint_pos = []
        for i in range(self.model.njnt):
            name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, i)
            angle = None
            if i < self.model.njnt - 1:
                angle = joint_positions[self.model.jnt_qposadr[i]:self.model.jnt_qposadr[i+1]]
            else:
                angle = joint_positions[self.model.jnt_qposadr[i]:]

            joint_pos.append([name, angle])

        return joint_pos
    
    def get_joint_vec(self):
        joint_vec = self.data.qvel
        return joint_vec

    def render(self):
        """ 渲染当前帧 """
        if self.viewer is None:
            self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
            self.viewer.render_every_frame = True
            self.viewer.cam.fixedcamid = 0  # 设置相机 ID
            # self.viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED  # 设为固定相机模式
        if self.viewer.is_running():
            self.viewer.sync()

    def quit(self):
        """ 退出仿真环境 """
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None

    def position_ctrl(self, actuator_id, target_pos):
        """ 控制指定驱动器(ACTUATOR_ID)到目标位置 """
        if actuator_id < 0 or actuator_id >= self.model.nu:
            raise ValueError(f"Invalid actuator ID: {actuator_id}")

        self.data.ctrl[actuator_id] = np.clip(target_pos, self.model.actuator_ctrlrange[actuator_id, 0], self.model.actuator_ctrlrange[actuator_id, 1])

    def position_set(self, joint_id, new_pos):
        """直接设置当前位置,new_pos为列表形式的角度向量"""
        if joint_id < 0 or joint_id >= self.model.njnt:
            raise ValueError(f"Invalid joint ID: {joint_id}")
        if joint_id < self.model.njnt - 1:
            self.data.qpos[self.model.jnt_qposadr[joint_id]:self.model.jnt_qposadr[joint_id+1]] = new_pos
        else:
            self.data.qpos[self.model.jnt_qposadr[joint_id]:] = new_pos


if __name__ == "__main__":
    env = MujocoSimEnv("assets\mjcx\scene.xml")
    env.reset()
    env.render()

    for _ in range(100000):
        # env.position_ctrl(2, 1.3)
        # env.position_set(3,[0.8, 0.5, 0, 0])
        # env.position_set(4,[0.8])
        # print(env.joint_vec)
        env.step()
        env.render()
        time.sleep(0.001)

    env.quit()