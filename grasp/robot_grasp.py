import time
import math
import threading
import numpy as np
from .helpers.matrix_funcs import euler2mat, convert_pose
from xarm.wrapper import XArmAPI


# 3次滑动平均，每组输入为4个变量。[x,y,z,yaw]
class Averager():
    def __init__(self, inputs, time_steps):
        self.buffer = np.zeros((time_steps, inputs))
        self.steps = time_steps
        self.curr = 0
        self.been_reset = True

    def update(self, v):
        if self.steps == 1:
            self.buffer = v
            return v
        self.buffer[self.curr, :] = v
        self.curr += 1
        if self.been_reset:
            self.been_reset = False
            while self.curr != 0:
                self.update(v)
        if self.curr >= self.steps:
            self.curr = 0
        pos = self.buffer.mean(axis=0)
        return pos

    def evaluate(self):
        if self.steps == 1:
            return self.buffer
        return self.buffer.mean(axis=0)

    def reset(self):
        self.buffer *= 0
        self.curr = 0
        self.been_reset = True


class MinPos():
    def __init__(self, inputs, time_steps):
        self.buffer = np.zeros((time_steps, inputs))
        self.steps = time_steps
        self.curr = 0
        self.been_reset = True
        self.inputs = inputs
        self.prev_pos = [0, 0, 0, 0]

    def update(self, v):
        if self.steps == 1:
            self.buffer = v
            return v
        self.buffer[self.curr, :] = v
        self.curr += 1
        if self.been_reset:
            self.been_reset = False
            while self.curr != 0:
                self.update(v)
        if self.curr >= self.steps:
            self.curr = 0
        min_inx = 0
        min_dis = 9999
        for i in range(self.steps):
            dis = pow(self.buffer[i][0] - self.prev_pos[0], 2) + pow(self.buffer[i][1] - self.prev_pos[1], 2) + pow(self.buffer[i][2] - self.prev_pos[2], 2)
            if dis < min_dis:
                min_dis = dis
                min_inx = i
        self.prev_pos[0] = self.buffer[min_inx][0]
        self.prev_pos[1] = self.buffer[min_inx][1]
        self.prev_pos[2] = self.buffer[min_inx][2]
        self.prev_pos[3] = self.buffer[min_inx][3]
        return self.prev_pos
    
    def evaluate(self):
        if self.steps == 1:
            return self.buffer
        return self.prev_pos

    def reset(self):
        self.buffer *= 0
        self.curr = 0
        self.prev = 0
        self.been_reset = True


class RobotGrasp(object):    
    CURR_POS = [300, 0, 350, 180, 0, 0]
    GOAL_POS = [0, 0, 0, 0, 0, 0]
    GRASP_STATUS = 0

    def __init__(self, robot_ip, ggcnn_cmd_que, euler_opt, grasp_config, click_mode=False):
        self.click_mode = click_mode
        self.arm = XArmAPI(robot_ip, report_type='real')
        self.ggcnn_cmd_que = ggcnn_cmd_que
        self.euler_eef_to_color_opt = euler_opt['EULER_EEF_TO_COLOR_OPT']
        self.euler_eef_to_color_opt2 = self.euler_eef_to_color_opt.copy()
        self.euler_eef_to_color_opt2[0] = 0
        self.euler_eef_to_color_opt2[1] = 0
        self.euler_color_to_depth_opt = euler_opt['EULER_COLOR_TO_DEPTH_OPT']
        self.grasping_range = grasp_config['GRASPING_RANGE']
        self.detect_xyz = grasp_config['DETECT_XYZ']
        self.release_xyz = grasp_config['RELEASE_XYZ']
        self.lift_height = self.detect_xyz[2] + grasp_config['LIFT_OFFSET_Z']
        self.gripper_z_mm = grasp_config['GRIPPER_Z_MM']
        self.grasping_min_z = grasp_config['GRASPING_MIN_Z']
        self.use_vacuum_gripper = grasp_config.get('USE_VACUUM_GRIPPER', False)
        self.lock_rpy = grasp_config.get('LOCK_RPY', False)
        self.min_result_z = grasp_config.get('MIN_RESULT_Z_MM', 200) / 1000
        self.pose_averager = MinPos(4, 3)
        self.pose_averager2 = MinPos(4, 3)
        self.ready_check = False
        self.ready_grasp = False
        self.alive = True
        self.last_grasp_time = 0
        self.is_over_range = False
        self.pos_t = threading.Thread(target=self.update_pos_loop, daemon=True)
        self.pos_t.start()
        self.ggcnn_t = threading.Thread(target=self.handle_ggcnn_loop, daemon=True)
        self.ggcnn_t.start()
        self.check_t = threading.Thread(target=self.check_loop, daemon=True)
        self.check_t.start()
    
    def is_alive(self):
        return self.alive
    
    def handle_ggcnn_loop(self):
        while self.arm.connected and self.alive:
            cmd = self.ggcnn_cmd_que.get()
            if self.click_mode:
                self.click_pick_place(cmd)
            else:
                self.grasp(cmd)

    def update_pos_loop(self):
        self.arm.motion_enable(True)
        self.arm.clean_error()
        self.arm.set_mode(0)
        self.arm.set_state(0)
        time.sleep(0.5)
        self.arm.set_position(z=self.detect_xyz[2], wait=True)
        self.arm.set_position(x=self.detect_xyz[0], y=self.detect_xyz[1], z=self.detect_xyz[2], roll=180, pitch=0, yaw=0, wait=True)
        time.sleep(0.5)

        if not self.use_vacuum_gripper:
            self.arm.set_gripper_enable(True)
        self.place()

        time.sleep(0.5)

        self.ready_grasp = True

        self.arm.set_mode(7)
        self.arm.set_state(0)
        time.sleep(0.5)

        self.ready_check = True

        while self.arm.connected:
            if self.arm.error_code != 0:
                print('[RECOVERY] ERROR_CODE: {}, attempting recovery...'.format(self.arm.error_code))
                self.ready_grasp = False
                self.ready_check = False
                try:
                    self.arm.clean_error()
                    self.arm.clean_warn()
                    time.sleep(0.5)
                    self.arm.motion_enable(True)
                    self.arm.set_mode(0)
                    self.arm.set_state(0)
                    time.sleep(1)
                    self.arm.set_position(z=self.detect_xyz[2], speed=100, wait=True)
                    self.arm.set_position(x=self.detect_xyz[0], y=self.detect_xyz[1], z=self.detect_xyz[2],
                                          roll=180, pitch=0, yaw=0, speed=100, wait=True)
                    time.sleep(0.5)
                    if not self.use_vacuum_gripper:
                        self.arm.set_gripper_enable(True)
                    self.place()
                    time.sleep(0.5)
                    self.pose_averager.reset()
                    self.arm.set_mode(7)
                    self.arm.set_state(0)
                    time.sleep(0.5)
                    self.ready_check = True
                    self.ready_grasp = True
                    self.GRASP_STATUS = 0
                    print('[RECOVERY] 恢复成功，继续运行')
                except Exception as e:
                    print('[RECOVERY] 恢复失败: {}'.format(e))
                    break
            else:
                _, pos = self.arm.get_position()
                self.arm.get_err_warn_code()
                self.CURR_POS = [pos[0], pos[1], pos[2], pos[3], pos[4], pos[5]]
                time.sleep(0.01)
        self.alive = False
        if self.arm.error_code != 0:
            print('ERROR_CODE: {}'.format(self.arm.error_code))
        print('*************** PROGRAM OVER *******************')
    
        self.arm.disconnect()
    
    def check_loop(self):
        while self.arm.connected and self.arm.error_code == 0:
            self._check()
            time.sleep(0.01)

    def _check(self):
        if not self.ready_check or not self.alive:
            return
        if self.click_mode:
            return
        x = self.CURR_POS[0]
        y = self.CURR_POS[1]
        z = self.CURR_POS[2]
        roll = self.CURR_POS[3]
        pitch = self.CURR_POS[4]
        yaw = self.CURR_POS[5]
        # reset to start position if moved outside of boundary
        if (time.monotonic() - self.last_grasp_time) > 5 or x < self.grasping_range[0] or x > self.grasping_range[1] or y < self.grasping_range[2] or y > self.grasping_range[3]:
            if (time.monotonic() - self.last_grasp_time) > 5 \
                and abs(x-self.detect_xyz[0]) < 2 and abs(y-self.detect_xyz[1]) < 2 and abs(z-self.detect_xyz[2]) < 2 \
                and abs(abs(roll)-180) < 2 and abs(pitch) < 2 and abs(yaw) < 2:
                self.last_grasp_time = time.monotonic()
                return
            print('[STOP] MOVE TO INITIAL DETECT POINT, CURR_POS={}, last_grasp_time={}'.format(self.CURR_POS, self.last_grasp_time))
            self.ready_grasp = False
            self.arm.set_state(4)
            self.arm.set_mode(0)
            self.arm.set_state(0)
            time.sleep(1)
            self.arm.set_position(z=self.detect_xyz[2], speed=200, wait=True)
            self.arm.set_position(x=self.detect_xyz[0], y=self.detect_xyz[1], z=self.detect_xyz[2], roll=180, pitch=0, yaw=0, speed=200, wait=True)
            time.sleep(0.25)
            self.pose_averager.reset()
            self.arm.set_mode(7)
            self.arm.set_state(0)
            time.sleep(0.5)
            self.GRASP_STATUS = 0
            self.ready_grasp = True
            self.last_grasp_time = time.monotonic()
            return

        if abs(self.CURR_POS[0] - self.GOAL_POS[0]) < 5 and abs(self.CURR_POS[1] - self.GOAL_POS[1]) < 5:
            if self.GRASP_STATUS == 0:
                self.GRASP_STATUS = 1
            elif self.GRASP_STATUS == 2:
                self.GRASP_STATUS = 3
                self.arm.set_position(*self.GOAL_POS, speed=50, acc=1000, wait=False)

        # Stop Conditions.
        if z < self.gripper_z_mm or z - 1 < self.GOAL_POS[2]:
            if not self.ready_grasp:
                return
            self.ready_grasp = False
            self.GRASP_STATUS = 0
            print('[GRASP] CURR_POS={}'.format(self.CURR_POS))
            self.arm.set_state(4)
            self.arm.set_mode(0)
            self.arm.set_state(0)
            time.sleep(0.1)

            self.pick()
            time.sleep(0.5)

            self.arm.set_position(z=self.lift_height, speed=200, wait=True)
            self.arm.set_position(x=self.release_xyz[0], y=self.release_xyz[1], roll=180, pitch=0, yaw=0, speed=200, wait=True)
            self.arm.set_position(z=self.release_xyz[2], speed=100, wait=True)

            self.place()

            self.arm.set_position(z=self.lift_height, speed=100, wait=True)
            self.arm.set_position(x=self.detect_xyz[0], y=self.detect_xyz[1], z=self.detect_xyz[2], roll=180, pitch=0, yaw=0, speed=200, wait=True)

            self.pose_averager.reset()
            self.arm.set_mode(7)
            self.arm.set_state(0)
            time.sleep(2)

            self.ready_grasp = True
            self.last_grasp_time = time.monotonic()

    def click_pick_place(self, data):
        """直接执行点击抓取放置"""
        if not self.alive or not self.ready_grasp:
            return

        euler_base_to_eef = data[0]
        d = list(data[1])

        if d[2] <= self.min_result_z:
            print('[CLICK] 深度值太小，跳过')
            return

        gp = [d[0], d[1], d[2], 0, 0, -1 * d[3]]

        mat_depthOpt_in_base = euler2mat(euler_base_to_eef) * euler2mat(self.euler_eef_to_color_opt) * euler2mat(self.euler_color_to_depth_opt)
        gp_base = convert_pose(gp, mat_depthOpt_in_base)

        if gp_base[5] < -np.pi:
            gp_base[5] += np.pi
        elif gp_base[5] > 0:
            gp_base[5] -= np.pi

        ang = gp_base[5] - np.pi / 2

        target_x = gp_base[0] * 1000
        target_y = gp_base[1] * 1000
        target_z = gp_base[2] * 1000 + self.gripper_z_mm
        target_z = max(target_z, self.grasping_min_z)

        if self.lock_rpy:
            target_yaw = 0
        else:
            target_yaw = math.degrees(ang + np.pi)

        if target_x < self.grasping_range[0] or target_x > self.grasping_range[1] or \
           target_y < self.grasping_range[2] or target_y > self.grasping_range[3]:
            print('[CLICK] 目标超出范围: [{:.1f}, {:.1f}, {:.1f}]'.format(target_x, target_y, target_z))
            return

        # 根据模型预测的宽度计算夹爪张开量
        predicted_width = d[4] if len(d) > 4 else 150
        GRIPPER_MARGIN = 2.0
        gripper_open_pos = min(850, max(400, int(predicted_width * GRIPPER_MARGIN / 150.0 * 850)))

        print('[CLICK] 抓取目标: [{:.1f}, {:.1f}, {:.1f}, yaw={:.1f}, 夹爪张开={:d}]'.format(
            target_x, target_y, target_z, target_yaw, gripper_open_pos))

        self.ready_grasp = False

        # 切换到位置控制模式
        self.arm.set_state(4)
        self.arm.set_mode(0)
        self.arm.set_state(0)
        time.sleep(0.5)

        # 0. 先张开夹爪到合适宽度
        if not self.use_vacuum_gripper:
            self.arm.set_gripper_position(gripper_open_pos, wait=True)
            time.sleep(0.3)

        # 1. 移到目标上方
        safe_z = max(target_z + 100, self.detect_xyz[2])
        self.arm.set_position(x=target_x, y=target_y, z=safe_z,
                              roll=180, pitch=0, yaw=target_yaw,
                              speed=200, wait=True)

        # 2. 下降到抓取高度
        self.arm.set_position(z=target_z, speed=80, wait=True)

        # 3. 抓取（闭合夹爪）
        self.pick()
        time.sleep(0.8)

        # 检查夹爪是否抓到东西
        grabbed = True
        if not self.use_vacuum_gripper:
            code, gripper_pos = self.arm.get_gripper_position()
            if code == 0 and gripper_pos is not None:
                print('[CLICK] 夹爪位置: {:.1f}'.format(gripper_pos))
                if gripper_pos < 5:
                    grabbed = False
                    print('[CLICK] 夹爪全闭合，未抓到物体，跳过递送')

        # 4. 抬起
        self.arm.set_position(z=self.lift_height, speed=200, wait=True)

        if grabbed:
            # 5. 移到释放位置
            self.arm.set_position(x=self.release_xyz[0], y=self.release_xyz[1],
                                  roll=180, pitch=0, yaw=0,
                                  speed=200, wait=True)
            self.arm.set_position(z=self.release_xyz[2], speed=100, wait=True)

            # 6. 释放
            self.place()
            time.sleep(0.5)
        else:
            self.place()
            time.sleep(0.3)

        # 7. 返回检测位
        self.arm.set_position(z=self.lift_height, speed=100, wait=True)
        self.arm.set_position(x=self.detect_xyz[0], y=self.detect_xyz[1], z=self.detect_xyz[2],
                              roll=180, pitch=0, yaw=0,
                              speed=200, wait=True)

        # 8. 恢复在线模式
        self.pose_averager.reset()
        self.arm.set_mode(7)
        self.arm.set_state(0)
        time.sleep(0.5)

        self.ready_grasp = True
        self.last_grasp_time = time.monotonic()
        print('[CLICK] 抓取完成')

    def pick(self):
        if self.use_vacuum_gripper:
            self.arm.set_vacuum_gripper(on=True)
        else:
            self.arm.set_gripper_position(0, wait=True)

    def place(self):
        if self.use_vacuum_gripper:
            self.arm.set_vacuum_gripper(on=False)
        else:
            self.arm.set_gripper_position(850, wait=True)

    def get_eef_pose_m(self):
        _, eef_pos_mm = self.arm.get_position(is_radian=True)
        eef_pos_m = [eef_pos_mm[0]*0.001, eef_pos_mm[1]*0.001, eef_pos_mm[2]*0.001, eef_pos_mm[3], eef_pos_mm[4], eef_pos_mm[5]]
        return eef_pos_m

    def grasp(self, data):
        if not self.alive or not self.ready_check or not self.ready_grasp:
            return

        euler_base_to_eef = data[0]
        d = list(data[1])

        if d[2] > self.min_result_z:
            gp = [d[0], d[1], d[2], 0, 0, -1 * d[3]]

            mat_depthOpt_in_base = euler2mat(euler_base_to_eef) * euler2mat(self.euler_eef_to_color_opt) * euler2mat(self.euler_color_to_depth_opt)
            mat_depthOpt_in_base2 = euler2mat(euler_base_to_eef) * euler2mat(self.euler_eef_to_color_opt2) * euler2mat(self.euler_color_to_depth_opt)
            gp_base = convert_pose(gp, mat_depthOpt_in_base)
            gp_base2 = convert_pose(gp, mat_depthOpt_in_base2)

            if gp_base[5] < -np.pi:
                gp_base[5] += np.pi
            elif gp_base[5] > 0:
                gp_base[5] -= np.pi

            if gp_base2[5] < -np.pi:
                gp_base2[5] += np.pi
            elif gp_base2[5] > 0:
                gp_base2[5] -= np.pi

            av = self.pose_averager.update(np.array([gp_base[0], gp_base[1], gp_base[2], gp_base[5]]))
            av2 = self.pose_averager2.update(np.array([gp_base2[0], gp_base2[1], gp_base2[2], gp_base2[5]]))
            if self.GRASP_STATUS == 0:
                av = av2
        else:
            gp_base = [0] * 6
            av = self.pose_averager.evaluate()

        ang = av[3] - np.pi/2
        gp_base = [av[0], av[0], av[0], np.pi, 0, ang]

        if self.lock_rpy:
            GOAL_POS = [av[0] * 1000, av[1] * 1000, av[2] * 1000 + self.gripper_z_mm, 180, 0, 0]
        else:
            GOAL_POS = [av[0] * 1000, av[1] * 1000, av[2] * 1000 + self.gripper_z_mm, 180, 0, math.degrees(ang + np.pi)]
        if GOAL_POS[2] < self.grasping_min_z - 10:
            return
        GOAL_POS[2] = max(GOAL_POS[2], self.grasping_min_z)

        if GOAL_POS[0] < self.grasping_range[0] or GOAL_POS[0] > self.grasping_range[1] or GOAL_POS[1] < self.grasping_range[2] or GOAL_POS[1] > self.grasping_range[3]:
            if not self.is_over_range:
                print('[WARN] TARGET IS OVER RANGE:', GOAL_POS)
            self.is_over_range = True
            return

        self.is_over_range = False
        self.last_grasp_time = time.monotonic()

        if self.GRASP_STATUS == 0:
            self.GOAL_POS = GOAL_POS
            z = max(self.CURR_POS[2], 380)
            self.arm.set_position(x=self.GOAL_POS[0], y=self.GOAL_POS[1], z=z,
                                  roll=self.GOAL_POS[3], pitch=self.GOAL_POS[4], yaw=self.GOAL_POS[5], 
                                  speed=100, acc=1000, wait=False)

        elif self.GRASP_STATUS == 1:
            z = max(self.CURR_POS[2], 380)
            self.GOAL_POS = GOAL_POS
            self.arm.set_position(x=GOAL_POS[0], y=GOAL_POS[1], z=z, roll=GOAL_POS[3], pitch=GOAL_POS[4], yaw=GOAL_POS[5], speed=200, acc=1000, wait=False)
            self.GRASP_STATUS = 2
