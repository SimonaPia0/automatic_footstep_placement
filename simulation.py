import numpy as np
import dartpy as dart
import copy
from utils import *
import os
import ismpc
import footstep_planner
import inverse_dynamics as id
import filter
import foot_trajectory_generator as ftg
from logger import Logger
import argparse

class Hrp4Controller(dart.gui.osg.RealTimeWorldNode):
    def __init__(self, world, hrp4, scene1, scene2, scene3, traj1, traj2):
        super(Hrp4Controller, self).__init__(world)
        self.world = world
        self.hrp4 = hrp4
        self.scene1 = scene1
        self.scene2 = scene2
        self.scene3 = scene3
        self.traj1 = traj1
        self.traj2 = traj2
        self.lateral_push_dir = 0.0
        self.time = 0
        self.params = {
            'g': 9.81,
            'h': 0.72,
            'foot_size': 0.1,
            'step_height': 0.02,
            'ss_duration': 50,
            'ds_duration': 30,
            'world_time_step': world.getTimeStep(),
            'first_swing': 'rfoot',
            'µ': 0.5,
            'N': 100,
            'dof': self.hrp4.getNumDofs(),
        }
        self.params['eta'] = np.sqrt(self.params['g'] / self.params['h'])

        # robot links
        self.lsole = hrp4.getBodyNode('l_sole')
        self.rsole = hrp4.getBodyNode('r_sole')
        self.torso = hrp4.getBodyNode('torso')
        self.base  = hrp4.getBodyNode('body')

        for i in range(hrp4.getNumJoints()):
            joint = hrp4.getJoint(i)
            dim = joint.getNumDofs()

            # set floating base to passive, everything else to torque
            if   dim == 6: joint.setActuatorType(dart.dynamics.ActuatorType.PASSIVE)
            elif dim == 1: joint.setActuatorType(dart.dynamics.ActuatorType.FORCE)

        # set initial configuration
        initial_configuration = {'CHEST_P': 0., 'CHEST_Y': 0., 'NECK_P': 0., 'NECK_Y': 0., \
                                 'R_HIP_Y': 0., 'R_HIP_R': -3., 'R_HIP_P': -25., 'R_KNEE_P': 50., 'R_ANKLE_P': -25., 'R_ANKLE_R':  3., \
                                 'L_HIP_Y': 0., 'L_HIP_R':  3., 'L_HIP_P': -25., 'L_KNEE_P': 50., 'L_ANKLE_P': -25., 'L_ANKLE_R': -3., \
                                 'R_SHOULDER_P': 4., 'R_SHOULDER_R': -8., 'R_SHOULDER_Y': 0., 'R_ELBOW_P': -25., \
                                 'L_SHOULDER_P': 4., 'L_SHOULDER_R':  8., 'L_SHOULDER_Y': 0., 'L_ELBOW_P': -25.}

        for joint_name, value in initial_configuration.items():
            self.hrp4.setPosition(self.hrp4.getDof(joint_name).getIndexInSkeleton(), value * np.pi / 180.)

        # position the robot on the ground
        lsole_pos = self.lsole.getTransform(withRespectTo=dart.dynamics.Frame.World(), inCoordinatesOf=dart.dynamics.Frame.World()).translation()
        rsole_pos = self.rsole.getTransform(withRespectTo=dart.dynamics.Frame.World(), inCoordinatesOf=dart.dynamics.Frame.World()).translation()
        self.hrp4.setPosition(3, - (lsole_pos[0] + rsole_pos[0]) / 2.)
        self.hrp4.setPosition(4, - (lsole_pos[1] + rsole_pos[1]) / 2.)
        self.hrp4.setPosition(5, - (lsole_pos[2] + rsole_pos[2]) / 2.)

        # initialize state
        self.initial = self.retrieve_state()
        self.contact = 'lfoot' if self.params['first_swing'] == 'rfoot' else 'rfoot' # there is a dummy footstep
        self.desired = copy.deepcopy(self.initial)

        # selection matrix for redundant dofs
        redundant_dofs = [ \
            "NECK_Y", "NECK_P", \
            "R_SHOULDER_P", "R_SHOULDER_R", "R_SHOULDER_Y", "R_ELBOW_P", \
            "L_SHOULDER_P", "L_SHOULDER_R", "L_SHOULDER_Y", "L_ELBOW_P"]
        
        # initialize inverse dynamics
        self.id = id.InverseDynamics(self.hrp4, redundant_dofs)

        # initialize footstep planner
        if self.traj1:
             reference = [(0.1, 0.0, 0.0)] * 25
        
        if self.traj2:
            reference = [(0.1, 0., 0.2)] * 5 + [(0.1, 0., -0.1)] * 10 + [(0.1, 0., 0.)] * 10


        self.footstep_planner = footstep_planner.FootstepPlanner(
            reference,
            self.initial['lfoot']['pos'],
            self.initial['rfoot']['pos'],
            self.params
            )


        # initialize MPC controller
        self.mpc = ismpc.Ismpc(
            self.initial, 
            self.footstep_planner, 
            self.params
            )

        # initialize foot trajectory generator
        self.foot_trajectory_generator = ftg.FootTrajectoryGenerator(
            self.initial, 
            self.footstep_planner, 
            self.params
            )

        # initialize kalman filter
        A = np.identity(3) + self.params['world_time_step'] * self.mpc.A_lip
        B = self.params['world_time_step'] * self.mpc.B_lip
        d = np.zeros(9)
        d[7] = - self.params['world_time_step'] * self.params['g']
        H = np.identity(3)
        Q = block_diag(1., 1., 1.)
        R = block_diag(1e1, 1e2, 1e4)
        P = np.identity(3)
        x = np.array([self.initial['com']['pos'][0], self.initial['com']['vel'][0], self.initial['zmp']['pos'][0], \
                      self.initial['com']['pos'][1], self.initial['com']['vel'][1], self.initial['zmp']['pos'][1], \
                      self.initial['com']['pos'][2], self.initial['com']['vel'][2], self.initial['zmp']['pos'][2]])
        self.kf = filter.KalmanFilter(block_diag(A, A, A), \
                                      block_diag(B, B, B), \
                                      d, \
                                      block_diag(H, H, H), \
                                      block_diag(Q, Q, Q), \
                                      block_diag(R, R, R), \
                                      block_diag(P, P, P), \
                                      x)

        # initialize logger and plots
        self.logger = Logger(self.initial)
        self.logger.initialize_plot(frequency=10)
        # --- AGGIUNTA PER DISEGNO FOOTSTEPS (ORIGINALI E ATTUALI) ---
        from matplotlib.patches import Rectangle
        ax_xy = self.logger.ax_xy
        
        # 1. Disegna i passi ORIGINALI (quelli nominali di ismpc)
        # Usiamo il bordo tratteggiato e un'alfa più bassa
        for step in self.mpc.original_plan:
            x, y = step['pos'][0], step['pos'][1]
            theta = step['ang'][2]
            w, h = 0.20, 0.14
            rect_orig = Rectangle((x - w/2, y - h/2), w, h, angle=np.rad2deg(theta),
                                 rotation_point='center', linewidth=1, 
                                 edgecolor='blue', linestyle='--', facecolor='none', alpha=0.2)
            ax_xy.add_patch(rect_orig)

        # 2. Disegna i passi ATTUALI (quelli che possono variare)
        self.step_patches = [] # <--- NUOVA LISTA PER SALVARE I RETTANGOLI
        for step in self.footstep_planner.plan:
            x, y = step['pos'][0], step['pos'][1]
            theta = step['ang'][2]
            w, h = 0.20, 0.14
            rect_curr = Rectangle((x - w/2, y - h/2), w, h, angle=np.rad2deg(theta),
                                 rotation_point='center', linewidth=1.2, 
                                 edgecolor='gray', facecolor='none', alpha=0.5)
            ax_xy.add_patch(rect_curr)
            self.step_patches.append(rect_curr) # <--- SALVIAMO IL RIFERIMENTO
        # -----------------------------------------------------------
        
    def customPreStep(self):
        # create current and desired states
        self.current = self.retrieve_state()

        # update kalman filter
        u = np.array([self.desired['zmp']['vel'][0], self.desired['zmp']['vel'][1], self.desired['zmp']['vel'][2]])
        self.kf.predict(u)
        x_flt, _ = self.kf.update(np.array([self.current['com']['pos'][0], self.current['com']['vel'][0], self.current['zmp']['pos'][0], \
                                            self.current['com']['pos'][1], self.current['com']['vel'][1], self.current['zmp']['pos'][1], \
                                            self.current['com']['pos'][2], self.current['com']['vel'][2], self.current['zmp']['pos'][2]]))
        
        # update current state using kalman filter output
        self.current['com']['pos'][0] = x_flt[0]
        self.current['com']['vel'][0] = x_flt[1]
        self.current['zmp']['pos'][0] = x_flt[2]
        self.current['com']['pos'][1] = x_flt[3]
        self.current['com']['vel'][1] = x_flt[4]
        self.current['zmp']['pos'][1] = x_flt[5]
        self.current['com']['pos'][2] = x_flt[6]
        self.current['com']['vel'][2] = x_flt[7]
        self.current['zmp']['pos'][2] = x_flt[8]

        is_pushed = False

        if self.scene1:
            # 1. Recupera lo stato attuale del piano dei passi
            idx = self.footstep_planner.get_step_index_at_time(self.time)
            phase = self.footstep_planner.get_phase_at_time(self.time)
            start_time_passo = self.footstep_planner.get_start_time(idx)

            # 2. Rileva l'istante esatto dell'inizio SS (es. al passo 5)
            if (self.time == start_time_passo) and (phase == 'ss') and (idx == 5):
                self.push_active_timer = 150  # L'MPC resterà "morbido" per 1.5 secondi
                self.force_duration = 3       # La spinta fisica dura solo 0.03 secondi

            # 3. Coordinazione is_pushed
            # Inizializziamo a False se non esiste ancora il timer
            is_pushed = hasattr(self, 'push_active_timer') and self.push_active_timer > 0

            # 4. Applicazione della Forza Fisica
            if hasattr(self, 'force_duration') and self.force_duration > 0:
                force = np.array([150.0, 0.0, 0.0]) # Spinta di 150N
                self.base.addExtForce(force)
                self.force_duration -= 1

            # 5. Decremento del timer MPC
            if is_pushed:
                self.push_active_timer -= 1

        if self.scene2:
            # 1. Recupera lo stato attuale del piano dei passi (identico a Scene 1)
            idx = self.footstep_planner.get_step_index_at_time(self.time)
            phase = self.footstep_planner.get_phase_at_time(self.time)
            start_time_passo = self.footstep_planner.get_start_time(idx)

            # 2. Trigger: Applica la forza all'inizio della fase SS (es. al passo 5)
            if (self.time == start_time_passo) and (phase == 'ss') and (idx == 5):
                # Timer per l'MPC: indica per quanto tempo il controllore deve compensare
                self.push_active_timer_s2 = 150  
                # Durata fisica della spinta (0.1 secondi = 10 step se dt=0.01)
                self.force_duration_s2 = 10       

            # --- LOGICA DINAMICA PER LO SWING ---
                # Recuperiamo l'ID del piede che deve muoversi in questo step
                swing_foot = self.footstep_planner.plan[idx]['foot_id']
                
                if swing_foot == 'rfoot':
                    # Se oscilla il destro, spingiamo verso DESTRA (Y negativa)
                    self.lateral_push_dir = -10.0
                else:
                    # Se oscilla il sinistro, spingiamo verso SINISTRA (Y positiva)
                    self.lateral_push_dir = 10.0
                # ------------------------------------

            # 3. Verifica se la perturbazione è attiva (per l'MPC)
            is_pushed = hasattr(self, 'push_active_timer_s2') and self.push_active_timer_s2 > 0

            # 4. Applicazione della Forza Fisica al TORSO (mantenendo i valori della Scene 2)
            if hasattr(self, 'force_duration_s2') and self.force_duration_s2 > 0:
                push_force = np.array([15.0, self.lateral_push_dir, 0.0]) # Forza diagonale
                self.torso.addExtForce(push_force)
                self.force_duration_s2 -= 1

            # 5. Decremento del timer per la compensazione MPC
            if is_pushed:
                self.push_active_timer_s2 -= 1

        if self.scene3:
            # Recupera passo corrente
            idx = self.footstep_planner.get_step_index_at_time(self.time)
            phase = self.footstep_planner.get_phase_at_time(self.time)
            start_time_passo = self.footstep_planner.get_start_time(idx)

            # Scegli qui il passo su cui vuoi applicare la perturbazione
            push_step_idx = 5   # oppure 5, ma deve essere quello reale del video

            # Trigger della spinta: solo all'inizio della single support del passo scelto
            if (self.time == start_time_passo) and (phase == 'ss') and (idx == push_step_idx):
                self.push_active_timer_s2 = 150
                self.force_duration_s2 = 10

                # Piede associato al passo corrente
                step_foot = self.footstep_planner.plan[idx]['foot_id']

                # Direzione esterna:
                # lfoot -> +Y
                # rfoot -> -Y
                if step_foot == 'lfoot':
                    self.lateral_push_dir = -20.0
                elif step_foot == 'rfoot':
                    self.lateral_push_dir = +20.0
                else:
                    self.lateral_push_dir = 0.0

            # L'MPC resta in modalità push per un po'
            is_pushed = hasattr(self, 'push_active_timer_s2') and self.push_active_timer_s2 > 0

            # La forza fisica dura pochi step
            if hasattr(self, 'force_duration_s2') and self.force_duration_s2 > 0:
                push_force = np.array([0.0, self.lateral_push_dir, 0.0])
                self.torso.addExtForce(push_force)
                self.force_duration_s2 -= 1

            if is_pushed:
                self.push_active_timer_s2 -= 1
        
        # Salviamo il CoM originale/desiderato PRIMA dell'MPC
        self.logger.log['desired', 'com_pure', 'pos'] = self.logger.log.get(('desired', 'com_pure', 'pos'), [])
        self.logger.log['desired', 'com_pure', 'pos'].append(copy.deepcopy(self.desired['com']['pos']))
        


        # get references using mpc
        lip_state, contact = self.mpc.solve(self.current, self.time, push_active=is_pushed)

        if is_pushed:
            idx_corr = self.footstep_planner.get_step_index_at_time(self.time)
            target_idx_corr = idx_corr + 1
            
            if target_idx_corr < len(self.footstep_planner.plan):
                piede_in_arrivo = self.footstep_planner.plan[target_idx_corr]['foot_id']
                nom_y = self.mpc.original_plan[target_idx_corr]['pos'][1] 
                curr_y = self.footstep_planner.plan[target_idx_corr]['pos'][1]
                spostamento_y = curr_y - nom_y
                
                print(f"[MPC RISOLTO] t={self.time} | Piede: {piede_in_arrivo} | "
                      f"Nominale: {nom_y:.4f} m -> Attuale: {curr_y:.4f} m | "
                      f"Spostamento: {spostamento_y:.4f} m")

        self.desired['com']['pos'] = lip_state['com']['pos']
        self.desired['com']['vel'] = lip_state['com']['vel']
        self.desired['com']['acc'] = lip_state['com']['acc']
        self.desired['zmp']['pos'] = lip_state['zmp']['pos']
        self.desired['zmp']['vel'] = lip_state['zmp']['vel']

        # get foot trajectories
        feet_trajectories = self.foot_trajectory_generator.generate_feet_trajectories_at_time(self.time)
        for foot in ['lfoot', 'rfoot']:
            for key in ['pos', 'vel', 'acc']:
                self.desired[foot][key] = feet_trajectories[foot][key]

        # set torso and base references to the average of the feet
        for link in ['torso', 'base']:
            for key in ['pos', 'vel', 'acc']:
                self.desired[link][key] = (self.desired['lfoot'][key][:3] + self.desired['rfoot'][key][:3]) / 2.

        # get torque commands using inverse dynamics
        commands = self.id.get_joint_torques(self.desired, self.current, contact) 
        
        # set acceleration commands
        for i in range(self.params['dof'] - 6):
            self.hrp4.setCommand(i + 6, commands[i])

        # In simulation.py, dentro customPreStep alla fine
        idx = self.footstep_planner.get_step_index_at_time(self.time)
        orig_step = self.mpc.original_plan[idx]
        target_foot = orig_step['foot_id'] # Identifica quale piede è nel piano originale

        for foot in ['lfoot', 'rfoot']:
            if foot == target_foot:
                # Registriamo la posizione del piede pianificato per questo step
                self.logger.log['original_foot', foot, 'pos'].append(orig_step['pos'])
            else:
                # Per l'altro piede (quello fermo), registriamo la posizione del passo precedente
                # o semplicemente manteniamo la coerenza nel grafico
                prev_idx = max(0, idx - 1)
                self.logger.log['original_foot', foot, 'pos'].append(self.mpc.original_plan[prev_idx]['pos'])

        # --- AGGIORNA LA POSIZIONE DEI RETTANGOLI GRIGI IN TEMPO REALE ---
        for i, step in enumerate(self.footstep_planner.plan):
            if i < len(self.step_patches):
                x, y = step['pos'][0], step['pos'][1]
                w, h = 0.20, 0.14
                
                # Aggiorniamo le coordinate (x,y) e l'angolo del rettangolo sul grafico
                self.step_patches[i].set_xy((x - w/2, y - h/2))
                self.step_patches[i].set_angle(np.rad2deg(step['ang'][2]))
        # -----------------------------------------------------------------

        # log and plot
        self.logger.log_data(self.current, self.desired)
        self.logger.update_plot(self.time)

        self.time += 1

    def retrieve_state(self):
        # com and torso pose (orientation and position)
        com_position = self.hrp4.getCOM()
        torso_orientation = get_rotvec(self.hrp4.getBodyNode('torso').getTransform(withRespectTo=dart.dynamics.Frame.World(), inCoordinatesOf=dart.dynamics.Frame.World()).rotation())
        base_orientation  = get_rotvec(self.hrp4.getBodyNode('body' ).getTransform(withRespectTo=dart.dynamics.Frame.World(), inCoordinatesOf=dart.dynamics.Frame.World()).rotation())

        # feet poses (orientation and position)
        l_foot_transform = self.lsole.getTransform(withRespectTo=dart.dynamics.Frame.World(), inCoordinatesOf=dart.dynamics.Frame.World())
        l_foot_orientation = get_rotvec(l_foot_transform.rotation())
        l_foot_position = l_foot_transform.translation()
        left_foot_pose = np.hstack((l_foot_orientation, l_foot_position))
        r_foot_transform = self.rsole.getTransform(withRespectTo=dart.dynamics.Frame.World(), inCoordinatesOf=dart.dynamics.Frame.World())
        r_foot_orientation = get_rotvec(r_foot_transform.rotation())
        r_foot_position = r_foot_transform.translation()
        right_foot_pose = np.hstack((r_foot_orientation, r_foot_position))

        # velocities
        com_velocity = self.hrp4.getCOMLinearVelocity(relativeTo=dart.dynamics.Frame.World(), inCoordinatesOf=dart.dynamics.Frame.World())
        torso_angular_velocity = self.hrp4.getBodyNode('torso').getAngularVelocity(relativeTo=dart.dynamics.Frame.World(), inCoordinatesOf=dart.dynamics.Frame.World())
        base_angular_velocity = self.hrp4.getBodyNode('body').getAngularVelocity(relativeTo=dart.dynamics.Frame.World(), inCoordinatesOf=dart.dynamics.Frame.World())
        l_foot_spatial_velocity = self.lsole.getSpatialVelocity(relativeTo=dart.dynamics.Frame.World(), inCoordinatesOf=dart.dynamics.Frame.World())
        r_foot_spatial_velocity = self.rsole.getSpatialVelocity(relativeTo=dart.dynamics.Frame.World(), inCoordinatesOf=dart.dynamics.Frame.World())

        # compute total contact force
        force = np.zeros(3)
        for contact in self.world.getLastCollisionResult().getContacts():
            force += contact.force

        # compute zmp
        zmp = np.zeros(3)
        zmp[2] = com_position[2] - force[2] / (self.hrp4.getMass() * self.params['g'] / self.params['h'])
        for contact in self.world.getLastCollisionResult().getContacts():
            if contact.force[2] <= 0.1: continue
            zmp[0] += (contact.point[0] * contact.force[2] / force[2] + (zmp[2] - contact.point[2]) * contact.force[0] / force[2])
            zmp[1] += (contact.point[1] * contact.force[2] / force[2] + (zmp[2] - contact.point[2]) * contact.force[1] / force[2])

        if force[2] <= 0.1: # threshold for when we lose contact
            zmp = np.array([0., 0., 0.]) # FIXME: this should return previous measurement
        else:
            # sometimes we get contact points that dont make sense, so we clip the ZMP close to the robot
            midpoint = (l_foot_position + r_foot_position) / 2.
            zmp[0] = np.clip(zmp[0], midpoint[0] - 0.3, midpoint[0] + 0.3)
            zmp[1] = np.clip(zmp[1], midpoint[1] - 0.3, midpoint[1] + 0.3)
            zmp[2] = np.clip(zmp[2], midpoint[2] - 0.3, midpoint[2] + 0.3)

        # create state dict
        return {
            'lfoot': {'pos': left_foot_pose,
                      'vel': l_foot_spatial_velocity,
                      'acc': np.zeros(6)},
            'rfoot': {'pos': right_foot_pose,
                      'vel': r_foot_spatial_velocity,
                      'acc': np.zeros(6)},
            'com'  : {'pos': com_position,
                      'vel': com_velocity,
                      'acc': np.zeros(3)},
            'torso': {'pos': torso_orientation,
                      'vel': torso_angular_velocity,
                      'acc': np.zeros(3)},
            'base' : {'pos': base_orientation,
                      'vel': base_angular_velocity,
                      'acc': np.zeros(3)},
            'joint': {'pos': self.hrp4.getPositions(),
                      'vel': self.hrp4.getVelocities(),
                      'acc': np.zeros(self.params['dof'])},
            'zmp'  : {'pos': zmp,
                      'vel': np.zeros(3),
                      'acc': np.zeros(3)}
        }

def main():
    world = dart.simulation.World()

    urdfParser = dart.utils.DartLoader()
    current_dir = os.path.dirname(os.path.abspath(__file__))
    hrp4   = urdfParser.parseSkeleton(os.path.join(current_dir, "urdf", "hrp4.urdf"))
    ground = urdfParser.parseSkeleton(os.path.join(current_dir, "urdf", "ground.urdf"))
    world.addSkeleton(hrp4)
    world.addSkeleton(ground)
    world.setGravity([0, 0, -9.81])
    world.setTimeStep(0.01)

    # set default inertia
    default_inertia = dart.dynamics.Inertia(1e-8, np.zeros(3), 1e-10 * np.identity(3))
    for body in hrp4.getBodyNodes():
        if body.getMass() == 0.0:
            body.setMass(1e-8)
            body.setInertia(default_inertia)

    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--scene', type=int)
    parser.add_argument('-t', '--traj', type=int )

    args = parser.parse_args()

    if args.scene == 1:
        scene1 = True
        scene2 = False
        scene3 = False
    elif args.scene == 2:
        scene1 = False
        scene2 = True
        scene3 = False
    elif args.scene == 3:
        scene1 = False
        scene2 = False
        scene3 = True
    else:
        scene1 = False
        scene2 = False
        scene3 = False

     
    if args.traj == 1:
        traj1 = True
        traj2 = False
    else:
        traj1 = False
        traj2 = True

    
    node = Hrp4Controller(world, hrp4, scene1, scene2,scene3,traj1,traj2)

    # create world node and add it to viewer
    viewer = dart.gui.osg.Viewer()
    node.setTargetRealTimeFactor(10) # speed up the visualization by 10x
    viewer.addWorldNode(node)

    #viewer.setUpViewInWindow(0, 0, 1920, 1080)
    viewer.setUpViewInWindow(0, 0, 1280, 720)
    #viewer.setUpViewInWindow(0, 0, 640, 480)
    viewer.setCameraHomePosition([5., -1., 1.5],
                                 [1.,  0., 0.5],
                                 [0.,  0., 1. ])
    try:
        viewer.run()
    finally:
        node.logger.close_video()

if __name__ == "__main__":
    main()