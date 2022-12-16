#!/usr/bin/env python

# Copyright (c) 2019: Jianyu Chen (jianyuchen@berkeley.edu)
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.


from __future__ import division

import copy
import numpy as np
import pygame
import random
import time
from skimage.transform import resize
import math

import gym
from gym import spaces
from gym.utils import seeding
import carla
import sys

from gym_carla.envs.render import BirdeyeRender
from gym_carla.envs.route_planner import RoutePlanner
from gym_carla.envs.misc import *


class CarlaEnv(gym.Env):
  """An OpenAI gym wrapper for CARLA simulator."""

  def __init__(self, params):
    global initial_task_mode
    global mode_change_interval
    global route
    global spawn_point
    global dest_point
    global dest_terminate
    global terminal_waypoint
    global episode_sum
    global current_quadrant_points
    
    # parameters
    self.display_size = params['display_size']  # rendering screen size
    self.max_past_step = params['max_past_step']
    self.number_of_vehicles = params['number_of_vehicles']
    self.number_of_walkers = params['number_of_walkers']
    self.dt = params['dt']
    self.task_mode = params['task_mode']
    initial_task_mode = params['task_mode']
    route = []
    mode_change_interval = 1 # if sequential_route_random_quadrants*{_sequential_point, _source_is_dest} or sequential_route_sequential_quadrant_*, must be 1.
    episode_sum = 3000 #元は5000
    dest_terminate = True
    spawn_point = None
    dest_point = None
    terminal_waypoint = None
    current_quadrant_points = []
    self.max_time_episode = params['max_time_episode']
    self.max_waypt = params['max_waypt']
    self.obs_range = params['obs_range']
    self.lidar_bin = params['lidar_bin']
    self.d_behind = params['d_behind']
    self.obs_size = int(self.obs_range/self.lidar_bin)
    self.out_lane_thres = params['out_lane_thres']
    self.desired_speed = params['desired_speed']
    self.max_ego_spawn_times = params['max_ego_spawn_times']
    self.display_route = params['display_route']
    if 'pixor' in params.keys():
      self.pixor = params['pixor']
      self.pixor_size = params['pixor_size']
    else:
      self.pixor = False

    # action and observation spaces
    self.discrete = params['discrete']
    self.discrete_act = [params['discrete_acc'], params['discrete_steer']] # acc, steer
    self.n_acc = len(self.discrete_act[0])
    self.n_steer = len(self.discrete_act[1])
    if self.discrete:
      self.action_space = spaces.Discrete(self.n_acc*self.n_steer)
    else:
      self.action_space = spaces.Box(np.array([params['continuous_accel_range'][0], 
      params['continuous_steer_range'][0]]), np.array([params['continuous_accel_range'][1],
      params['continuous_steer_range'][1]]), dtype=np.float32)  # acc, steer
    observation_space_dict = {
      'camera': spaces.Box(low=0, high=255, shape=(self.obs_size, self.obs_size, 3), dtype=np.uint8),
      'lidar': spaces.Box(low=0, high=255, shape=(self.obs_size, self.obs_size, 3), dtype=np.uint8),
      'birdeye': spaces.Box(low=0, high=255, shape=(self.obs_size, self.obs_size, 3), dtype=np.uint8),
      'state': spaces.Box(np.array([-2, -1, -5, 0]), np.array([2, 1, 30, 1]), dtype=np.float32)
      }
    if self.pixor:
      observation_space_dict.update({
        'roadmap': spaces.Box(low=0, high=255, shape=(self.obs_size, self.obs_size, 3), dtype=np.uint8),
        'vh_clas': spaces.Box(low=0, high=1, shape=(self.pixor_size, self.pixor_size, 1), dtype=np.float32),
        'vh_regr': spaces.Box(low=-5, high=5, shape=(self.pixor_size, self.pixor_size, 6), dtype=np.float32),
        'pixor_state': spaces.Box(np.array([-1000, -1000, -1, -1, -5]), np.array([1000, 1000, 1, 1, 20]), dtype=np.float32)
        })
    self.observation_space = spaces.Dict(observation_space_dict)

    # Connect to carla server and get world object
    print('connecting to Carla server...')
    client = carla.Client('localhost', params['port'])
    client.set_timeout(10.0)
    self.world = client.load_world(params['town'])
    print('Carla server connected!')

    # Set weather
    self.world.set_weather(carla.WeatherParameters.ClearNoon)

    # Get spawn points
    self.vehicle_spawn_points = list(self.world.get_map().get_spawn_points())
    self.finish_point = random.choice(self.vehicle_spawn_points).location
    self.walker_spawn_points = []
    for i in range(self.number_of_walkers):
      spawn_point = carla.Transform()
      loc = self.world.get_random_location_from_navigation()
      if (loc != None):
        spawn_point.location = loc
        self.walker_spawn_points.append(spawn_point)

    # Create the ego vehicle blueprint
    self.ego_bp = self._create_vehicle_bluepprint(params['ego_vehicle_filter'], color='49,8,8')

    # Collision sensor
    self.collision_hist = [] # The collision history
    self.collision_hist_l = 1 # collision history length
    self.collision_bp = self.world.get_blueprint_library().find('sensor.other.collision')

    # Lidar sensor
    self.lidar_data = None
    self.lidar_height = 2.1
    self.lidar_trans = carla.Transform(carla.Location(x=0.0, z=self.lidar_height))
    self.lidar_bp = self.world.get_blueprint_library().find('sensor.lidar.ray_cast')
    self.lidar_bp.set_attribute('channels', '32')
    self.lidar_bp.set_attribute('range', '5000')

    # Camera sensor
    self.camera_img = np.zeros((self.obs_size, self.obs_size, 3), dtype=np.uint8)
    self.camera_trans = carla.Transform(carla.Location(x=0.8, z=1.7))
    self.camera_bp = self.world.get_blueprint_library().find('sensor.camera.rgb')
    # Modify the attributes of the blueprint to set image resolution and field of view.
    self.camera_bp.set_attribute('image_size_x', str(self.obs_size))
    self.camera_bp.set_attribute('image_size_y', str(self.obs_size))
    self.camera_bp.set_attribute('fov', '110')
    # Set the time in seconds between sensor captures
    self.camera_bp.set_attribute('sensor_tick', '0.02')

    # Set fixed simulation step for synchronous mode
    self.settings = self.world.get_settings()
    self.settings.fixed_delta_seconds = self.dt

    # Record the time of total steps and resetting steps
    self.reset_step = 0
    self.total_step = 0
    
    # Initialize the renderer
    self._init_renderer()

    # Get pixel grid points
    if self.pixor:
      x, y = np.meshgrid(np.arange(self.pixor_size), np.arange(self.pixor_size)) # make a canvas with coordinates
      x, y = x.flatten(), y.flatten()
      self.pixel_grid = np.vstack((x, y)).T
    
    #avoid reset_loop
    self.reset_loop_flag=0

  def reset(self):
    
    print("Reset_Start")
    
    global initial_task_mode
    global mode_change_interval
    global route
    global spawn_point
    global dest_point
    global episode_sum
    global terminal_waypoint
    global col_terminate
    global max_step_terminate
    global dest_terminate
    global out_of_lane_terminate
    global current_quadrant_points

    # Grouping {spawn, dest} points for each quadrant
    first_quadrant_points=[]
    second_quadrant_points=[]
    third_quadrant_points=[]
    fourth_quadrant_points=[]
    count = 0
    for point in self.vehicle_spawn_points:
      count = count + 1
      if point.location.x >= 0 and point.location.y >= 0:
        first_quadrant_points.append(point)
        continue
      elif point.location.x <= 0 and point.location.y >= 0:
        second_quadrant_points.append(point)
        continue
      elif point.location.x <= 0 and point.location.y <= 0:
        third_quadrant_points.append(point)
        continue
      elif point.location.x >= 0 and point.location.y <= 0:
        fourth_quadrant_points.append(point)
        continue
    
    # Clear sensor objects  
    self.collision_sensor = None
    self.lidar_sensor = None
    self.camera_sensor = None

    # Delete sensors, vehicles and walkers
    self._clear_all_actors(['sensor.other.collision', 'sensor.lidar.ray_cast', 'sensor.camera.rgb', 'vehicle.*', 'controller.ai.walker', 'walker.*'])

    # Disable sync mode
    self._set_synchronous_mode(False)

    # Spawn surrounding vehicles
    random.shuffle(self.vehicle_spawn_points)
    count = self.number_of_vehicles
    if count > 0:
      for spawn_point in self.vehicle_spawn_points:
        if self._try_spawn_random_vehicle_at(spawn_point, number_of_wheels=[4]):
          count -= 1
        if count <= 0:
          break
    while count > 0:
      if self._try_spawn_random_vehicle_at(random.choice(self.vehicle_spawn_points), number_of_wheels=[4]):
        count -= 1

    # Spawn pedestrians
    random.shuffle(self.walker_spawn_points)
    count = self.number_of_walkers
    if count > 0:
      for spawn_point in self.walker_spawn_points:
        if self._try_spawn_random_walker_at(spawn_point):
          count -= 1
        if count <= 0:
          break
    while count > 0:
      if self._try_spawn_random_walker_at(random.choice(self.walker_spawn_points)):
        count -= 1

    # Get actors polygon list
    self.vehicle_polygons = []
    vehicle_poly_dict = self._get_actor_polygons('vehicle.*')
    self.vehicle_polygons.append(vehicle_poly_dict)
    self.walker_polygons = []
    walker_poly_dict = self._get_actor_polygons('walker.*')
    self.walker_polygons.append(walker_poly_dict)

    # Spawn the ego vehicle
    ego_spawn_times = 0

    route_sets = [['first_quadrant', 'second_quadrant'],['second_quadrant', 'first_quadrant']]#town2はマップが小さい
    #[['first_quadrant', 'second_quadrant'], ['second_quadrant', 'third_quadrant'], ['third_quadrant', 'fourth_quadrant'], ['fourth_quadrant', 'first_quadrant'], ['second_quadrant', 'first_quadrant'], ['third_quadrant', 'second_quadrant'], ['fourth_quadrant', 'third_quadrant'], ['first_quadrant', 'fourth_quadrant']] #[spawn, dest]

    #print("DEBUG: initial task mode is ", end="")
    #print(initial_task_mode)

    #Route update
    """
    if self.reset_step % mode_change_interval == 0: 
      if initial_task_mode == 'sequential_quadrants':
        if self.reset_step % (4 * mode_change_interval) == 0:
          route = ['first_quadrant', 'first_quadrant']
        elif self.reset_step % (4 * mode_change_interval) == 1 * mode_change_interval:
          route = ['second_quadrant', 'second_quadrant']
        elif self.reset_step % (4 * mode_change_interval) == 2 * mode_change_interval:
          route = ['third_quadrant', 'third_quadrant']
        elif self.reset_step % (4 * mode_change_interval) == 3 * mode_change_interval:
          route = ['fourth_quadrant', 'fourth_quadrant']
      elif initial_task_mode == 'random_quadrants':
        quadrant = random.choice(['first_quadrant', 'second_quadrant', 'third_quadrant', 'fourth_quadrant'])
        route = []
        route.append(quadrant)
        route.append(quadrant)
      elif initial_task_mode == 'general_route_random_quadrants':
        route = random.choice(route_sets)
      elif initial_task_mode == 'sequential_route_duplicate_quadrants_random_points':
        if self.reset_step % (8 * mode_change_interval) == 0:
          route = ['first_quadrant', 'second_quadrant']
        elif self.reset_step % (8 * mode_change_interval) == 1 * mode_change_interval:
          route = ['second_quadrant', 'third_quadrant']
        elif self.reset_step % (8 * mode_change_interval) == 2 * mode_change_interval:
          route = ['second_quadrant', 'third_quadrant']
        elif self.reset_step % (8 * mode_change_interval) == 3 * mode_change_interval:
          route = ['third_quadrant', 'fourth_quadrant']
        elif self.reset_step % (8 * mode_change_interval) == 4 * mode_change_interval:
          route = ['third_quadrant', 'fourth_quadrant']
        elif self.reset_step % (8 * mode_change_interval) == 5 * mode_change_interval:
          route = ['fourth_quadrant', 'first_quadrant']
        elif self.reset_step % (8 * mode_change_interval) == 6 * mode_change_interval:
          route = ['fourth_quadrant', 'first_quadrant']
        elif self.reset_step % (8 * mode_change_interval) == 7 * mode_change_interval:
          route = ['first_quadrant', 'second_quadrant']       
      elif 'sequential_route' in initial_task_mode:
        if 'duplicate_quadrants' in initial_task_mode:
          if self.reset_step % (8 * mode_change_interval) == 0:
            route = ['first_quadrant', 'second_quadrant']
          elif self.reset_step % (8 * mode_change_interval) == 1 * mode_change_interval:
            route = ['second_quadrant', 'third_quadrant']
          elif self.reset_step % (8 * mode_change_interval) == 2 * mode_change_interval:
            route = ['second_quadrant', 'third_quadrant']
          elif self.reset_step % (8 * mode_change_interval) == 3 * mode_change_interval:
            route = ['third_quadrant', 'fourth_quadrant']
          elif self.reset_step % (8 * mode_change_interval) == 4 * mode_change_interval:
            route = ['third_quadrant', 'fourth_quadrant']
          elif self.reset_step % (8 * mode_change_interval) == 5 * mode_change_interval:
            route = ['fourth_quadrant', 'first_quadrant']
          elif self.reset_step % (8 * mode_change_interval) == 6 * mode_change_interval:
            route = ['fourth_quadrant', 'first_quadrant']
          elif self.reset_step % (8 * mode_change_interval) == 7 * mode_change_interval:
            route = ['first_quadrant', 'second_quadrant']       
        elif 'random_quadrants' in initial_task_mode:
          if len(route) == 0:
            #print("DEBUG: route is Null")
            route = random.choice(route_sets)
          elif len(route) > 1:
            #print("DEBUG: route is not Null")
            route[0] = copy.deepcopy(route[1])
            dest_sets = ['first_quadrant', 'second_quadrant', 'third_quadrant', 'fourth_quadrant']
            dest_sets.remove(route[1])
            if route[1] == 'first_quadrant':
              dest_sets.remove('third_quadrant')
              current_quadrant_points = first_quadrant_points
            elif route[1] == 'second_quadrant':
              dest_sets.remove('fourth_quadrant')
              current_quadrant_points = second_quadrant_points
            elif route[1] == 'third_quadrant':
              dest_sets.remove('first_quadrant')
              current_quadrant_points = third_quadrant_points
            elif route[1] == 'fourth_quadrant':
              dest_sets.remove('second_quadrant')
              current_quadrant_points = fourth_quadrant_points
              route[1] = random.choice(dest_sets)
        elif 'sequential_quadrants' in initial_task_mode:
          if len(route) == 0:
            #print("DEBUG: route is Null")
            route = ['first_quadrant', 'second_quadrant']
            current_quadrant_points = first_quadrant_points
          elif len(route) > 1:
            route[0] = copy.deepcopy(route[1])
            if route[0] == 'first_quadrant':
              current_quadrant_points = first_quadrant_points
              route[1] = 'second_quadrant'
            elif route[0] == 'second_quadrant':
              current_quadrant_points = second_quadrant_points
              route[1] = 'third_quadrant'
            elif route[0] == 'third_quadrant':
              current_quadrant_points = third_quadrant_points
              route[1] = 'fourth_quadrant'
            elif route[0] == 'fourth_quadrant':
              current_quadrant_points = fourth_quadrant_points
              route[1] = 'first_quadrant'
    """

    #set spawn point
    """
    if initial_task_mode == 'simple_route_quad1_to_quad2':
      self.task_mode = 'first_quadrant'
    elif initial_task_mode == 'simple_route_quad2_to_quad3':
      self.task_mode = 'second_quadrant'
    elif initial_task_mode == 'simple_route_quad3_to_quad4':
      self.task_mode = 'third_quadrant'
    elif initial_task_mode == 'simple_route_quad4_to_quad1':
      self.task_mode = 'fourth_quadrant'
    elif initial_task_mode == 'sequential_quadrants' or initial_task_mode == 'random_quadrants' or 'sequential_route' in initial_task_mode or 'general_route' in initial_task_mode:
      self.task_mode = route[0]
      if 'source_is_dest' in initial_task_mode:
        if dest_point != None:
          self.task_mode = 'source_is_dest'
      elif 'sequential_point' in initial_task_mode:
        if dest_point != None:
          if terminal_waypoint == None:
            self.task_mode = 'source_is_dest'
          else:
            self.task_mode = 'sequential_point'      
    
    while ego_spawn_times <= self.max_ego_spawn_times:
      ''' by Nakazima 2020/12/30
      if self.reset_step <=1000:
        spawn_point = random.choice(self.vehicle_spawn_points)
        
      if self.reset_step > 1000:
        self.start=[92.1,-4.2,178.66] # easy60
        #self.start=[62.1,-4.2, 178.66] # diffi
        #self.start=[52.1+np.random.uniform(-5,5),-4.2, 178.66] # random
        # self.start=[52.1,-4.2, 178.66] # static
        spawn_point = set_carla_transform(self.start)
      '''
      if self.task_mode == 'random':
        spawn_point = random.choice(self.vehicle_spawn_points)
      if self.task_mode == 'roundabout':
        self.start=[52.1+np.random.uniform(-5,5),-4.2, 178.66] # random
        # self.start=[52.1,-4.2, 178.66] # static
        spawn_point = set_carla_transform(self.start)
      if self.task_mode == 'first_quadrant':
        spawn_point = random.choice(first_quadrant_points)
        if '1_point_by_1_division' in initial_task_mode:
          spawn_point = get_closest_vehicle_spawn_point(current_quadrant_points, [105,63])
        elif '4_points_by_1_division' in initial_task_mode:
          spawn_points = []
          candidate = get_closest_vehicle_spawn_point(current_quadrant_points, [105,63])
          spawn_points.append(candidate)
          candidate = get_closest_vehicle_spawn_point(current_quadrant_points, [201,63])
          spawn_points.append(candidate)
          candidate = get_closest_vehicle_spawn_point(current_quadrant_points, [105,205])
          spawn_points.append(candidate)
          candidate = get_closest_vehicle_spawn_point(current_quadrant_points, [242,133])
          spawn_points.append(candidate)
          if dest_point == None:
            spawn_point = random.choice(spawn_points)
          else:
            spawn_point = get_closest_vehicle_spawn_point(current_quadrant_points, [dest_point.location.x, dest_point.location.y])
        elif '2_points_by_1_division' in initial_task_mode:
          spawn_points = []
          candidate = get_closest_vehicle_spawn_point(current_quadrant_points, [105,63])
          spawn_points.append(candidate)
          candidate = get_closest_vehicle_spawn_point(current_quadrant_points, [242,133])
          spawn_points.append(candidate)
          if dest_point == None:
            spawn_point = random.choice(spawn_points)
          else:
            spawn_point = get_closest_vehicle_spawn_point(current_quadrant_points, [dest_point.location.x, dest_point.location.y])
        elif '8_points_by_1_division' in initial_task_mode:
          spawn_points = []
          candidate = get_closest_vehicle_spawn_point(current_quadrant_points, [66,4])
          spawn_points.append(candidate)
          candidate = get_closest_vehicle_spawn_point(current_quadrant_points, [137,4])
          spawn_points.append(candidate)
          candidate = get_closest_vehicle_spawn_point(current_quadrant_points, [209,4])
          spawn_points.append(candidate)
          candidate = get_closest_vehicle_spawn_point(current_quadrant_points, [105,63])
          spawn_points.append(candidate)
          candidate = get_closest_vehicle_spawn_point(current_quadrant_points, [105,134])
          spawn_points.append(candidate)
          candidate = get_closest_vehicle_spawn_point(current_quadrant_points, [66,205])
          spawn_points.append(candidate)
          candidate = get_closest_vehicle_spawn_point(current_quadrant_points, [137,205])
          spawn_points.append(candidate)
          candidate = get_closest_vehicle_spawn_point(current_quadrant_points, [209,195])
          spawn_points.append(candidate)
          if dest_point == None:
            spawn_point = random.choice(spawn_points)
          else:
            spawn_point = get_closest_vehicle_spawn_point(current_quadrant_points, [dest_point.location.x, dest_point.location.y])
      if self.task_mode == 'second_quadrant':
        spawn_point = random.choice(second_quadrant_points)
        if '1_point_by_1_division' in initial_task_mode:
          spawn_point = get_closest_vehicle_spawn_point(current_quadrant_points, [-84,63])
        elif '4_points_by_1_division' in initial_task_mode:
          spawn_points = []
          candidate = get_closest_vehicle_spawn_point(current_quadrant_points, [-146,53])
          spawn_points.append(candidate)
          candidate = get_closest_vehicle_spawn_point(current_quadrant_points, [-84,63])
          spawn_points.append(candidate)
          candidate = get_closest_vehicle_spawn_point(current_quadrant_points, [-6,171])
          spawn_points.append(candidate)
          candidate = get_closest_vehicle_spawn_point(current_quadrant_points, [-84,165])
          spawn_points.append(candidate)
          if dest_point == None:
            spawn_point = random.choice(spawn_points)
          else:
            spawn_point = get_closest_vehicle_spawn_point(current_quadrant_points, [dest_point.location.x, dest_point.location.y])
        elif '2_points_by_1_division' in initial_task_mode:
          spawn_points = []
          candidate = get_closest_vehicle_spawn_point(current_quadrant_points, [-146,53])
          spawn_points.append(candidate)
          candidate = get_closest_vehicle_spawn_point(current_quadrant_points, [-6,171])
          spawn_points.append(candidate)
          if dest_point == None:
            spawn_point = random.choice(spawn_points)
          else:
            spawn_point = get_closest_vehicle_spawn_point(current_quadrant_points, [dest_point.location.x, dest_point.location.y])
        elif '8_points_by_1_division' in initial_task_mode:
          spawn_points = []
          candidate = get_closest_vehicle_spawn_point(current_quadrant_points, [-146,53])
          spawn_points.append(candidate)
          candidate = get_closest_vehicle_spawn_point(current_quadrant_points, [-75,53])
          spawn_points.append(candidate)
          candidate = get_closest_vehicle_spawn_point(current_quadrant_points, [-3,53])
          spawn_points.append(candidate)
          candidate = get_closest_vehicle_spawn_point(current_quadrant_points, [-146,124])
          spawn_points.append(candidate)
          candidate = get_closest_vehicle_spawn_point(current_quadrant_points, [-75,124])
          spawn_points.append(candidate)
          candidate = get_closest_vehicle_spawn_point(current_quadrant_points, [-3,124])
          spawn_points.append(candidate)
          candidate = get_closest_vehicle_spawn_point(current_quadrant_points, [-75,195])
          spawn_points.append(candidate)
          candidate = get_closest_vehicle_spawn_point(current_quadrant_points, [-3,195])
          spawn_points.append(candidate)
          if dest_point == None:
            spawn_point = random.choice(spawn_points)
          else:
            spawn_point = get_closest_vehicle_spawn_point(current_quadrant_points, [dest_point.location.x, dest_point.location.y])
      if self.task_mode == 'third_quadrant':
        spawn_point = random.choice(third_quadrant_points)
        if '1_point_by_1_division' in initial_task_mode:
          spawn_point = get_closest_vehicle_spawn_point(current_quadrant_points, [-84,-123])
        elif '4_points_by_1_division' in initial_task_mode:
          spawn_points = []
          candidate = get_closest_vehicle_spawn_point(current_quadrant_points, [-146,-90])
          spawn_points.append(candidate)
          candidate = get_closest_vehicle_spawn_point(current_quadrant_points, [-84,-123])
          spawn_points.append(candidate)
          candidate = get_closest_vehicle_spawn_point(current_quadrant_points, [-84,-52])
          spawn_points.append(candidate)
          candidate = get_closest_vehicle_spawn_point(current_quadrant_points, [-17,-140])
          spawn_points.append(candidate)
          if dest_point == None:
            spawn_point = random.choice(spawn_points)
          else:
            spawn_point = get_closest_vehicle_spawn_point(current_quadrant_points, [dest_point.location.x, dest_point.location.y])
        elif '2_points_by_1_division' in initial_task_mode:
          spawn_points = []
          candidate = get_closest_vehicle_spawn_point(current_quadrant_points, [-146,-90])
          spawn_points.append(candidate)
          candidate = get_closest_vehicle_spawn_point(current_quadrant_points, [-17,-140])
          spawn_points.append(candidate)
          if dest_point == None:
            spawn_point = random.choice(spawn_points)
          else:
            spawn_point = get_closest_vehicle_spawn_point(current_quadrant_points, [dest_point.location.x, dest_point.location.y])
        elif '8_points_by_1_division' in initial_task_mode:
          spawn_points = []
          candidate = get_closest_vehicle_spawn_point(current_quadrant_points, [-161,-10])
          spawn_points.append(candidate)
          candidate = get_closest_vehicle_spawn_point(current_quadrant_points, [-161,-82])
          spawn_points.append(candidate)
          candidate = get_closest_vehicle_spawn_point(current_quadrant_points, [-89,-10])
          spawn_points.append(candidate)
          candidate = get_closest_vehicle_spawn_point(current_quadrant_points, [-89,-82])
          spawn_points.append(candidate)
          candidate = get_closest_vehicle_spawn_point(current_quadrant_points, [-89,-153])
          spawn_points.append(candidate)
          candidate = get_closest_vehicle_spawn_point(current_quadrant_points, [-18,-36])
          spawn_points.append(candidate)
          candidate = get_closest_vehicle_spawn_point(current_quadrant_points, [-18,-107])
          spawn_points.append(candidate)
          candidate = get_closest_vehicle_spawn_point(current_quadrant_points, [-18,-179])
          spawn_points.append(candidate)
          if dest_point == None:
            spawn_point = random.choice(spawn_points)
          else:
            spawn_point = get_closest_vehicle_spawn_point(current_quadrant_points, [dest_point.location.x, dest_point.location.y])
      if self.task_mode == 'fourth_quadrant':
        spawn_point = random.choice(fourth_quadrant_points)
        if '1_point_by_1_division' in initial_task_mode:
          spawn_point = get_closest_vehicle_spawn_point(current_quadrant_points, [76,-123])
        elif '4_points_by_1_division' in initial_task_mode:
          spawn_points = []
          candidate = get_closest_vehicle_spawn_point(current_quadrant_points, [72,-201])
          spawn_points.append(candidate)
          candidate = get_closest_vehicle_spawn_point(current_quadrant_points, [178,-201])
          spawn_points.append(candidate)
          candidate = get_closest_vehicle_spawn_point(current_quadrant_points, [242,-10])
          spawn_points.append(candidate)
          candidate = get_closest_vehicle_spawn_point(current_quadrant_points, [68,-10])
          spawn_points.append(candidate)
          if dest_point == None:
            spawn_point = random.choice(spawn_points)
          else:
            spawn_point = get_closest_vehicle_spawn_point(current_quadrant_points, [dest_point.location.x, dest_point.location.y])
        elif '2_points_by_1_division' in initial_task_mode:
          spawn_points = []
          candidate = get_closest_vehicle_spawn_point(current_quadrant_points, [72,-201])
          spawn_points.append(candidate)
          candidate = get_closest_vehicle_spawn_point(current_quadrant_points, [242,-10])
          spawn_points.append(candidate)
          if dest_point == None:
            spawn_point = random.choice(spawn_points)
          else:
            spawn_point = get_closest_vehicle_spawn_point(current_quadrant_points, [dest_point.location.x, dest_point.location.y])
        elif '8_points_by_1_division' in initial_task_mode:
          spawn_points = []
          candidate = get_closest_vehicle_spawn_point(current_quadrant_points, [82,-201])
          spawn_points.append(candidate)
          candidate = get_closest_vehicle_spawn_point(current_quadrant_points, [82,-130])
          spawn_points.append(candidate)
          candidate = get_closest_vehicle_spawn_point(current_quadrant_points, [82,-58])
          spawn_points.append(candidate)
          candidate = get_closest_vehicle_spawn_point(current_quadrant_points, [143,-201])
          spawn_points.append(candidate)
          candidate = get_closest_vehicle_spawn_point(current_quadrant_points, [143,-130])
          spawn_points.append(candidate)
          candidate = get_closest_vehicle_spawn_point(current_quadrant_points, [143,-58])
          spawn_points.append(candidate)
          candidate = get_closest_vehicle_spawn_point(current_quadrant_points, [242,-117])
          spawn_points.append(candidate)
          candidate = get_closest_vehicle_spawn_point(current_quadrant_points, [242,-10])
          spawn_points.append(candidate)
          if dest_point == None:
            spawn_point = random.choice(spawn_points)
          else:
            spawn_point = get_closest_vehicle_spawn_point(current_quadrant_points, [dest_point.location.x, dest_point.location.y])
      if self.task_mode == 'source_is_dest':
        spawn_point = get_closest_vehicle_spawn_point(current_quadrant_points, [dest_point.location.x, dest_point.location.y])
      if self.task_mode == 'sequential_point':
        spawn_point = terminal_waypoint.transform
      if self._try_spawn_ego_vehicle_at(spawn_point):
        #print("DEBUG: Spawn Point is ", end="")
        #print(spawn_point.location)
        break
      else:
        if self.task_mode == 'source_is_dest' or '_division' in initial_task_mode:
          if len(current_quadrant_points) == 0:
            ego_spawn_times += self.max_ego_spawn_times + 1
          else:
            current_quadrant_points.remove(spawn_point)
        elif self.task_mode == 'sequential_point':
          '''
          next_waypoints = terminal_waypoint.next(1.0)
          distance = 1
          while len(next_waypoints) == 0:
            distancself.first_vehicle_spawn_flage += 1
            next_waypoints = terminal_waypoint.next(1.0*distance)
            if distance == self.max_ego_spawn_times:
              terminal_waypoint = None
              ego_spawn_times += self.max_ego_spawn_times + 1
              continue
          terminal_waypoint = random.choice(next_waypoints)
          ego_spawn_times += 1
          '''
          dest_terminate = True
          terminal_waypoint = None
          mode_change_interval = 1 #Update route
          ego_spawn_times += self.max_ego_spawn_times + 1
        else:
          ego_spawn_times += 1
        time.sleep(0.1)
    """
    
    
    #引き継ぎスポーン

    fin_x = self.finish_point.x
    fin_y = self.finish_point.y
    while ego_spawn_times <= self.max_ego_spawn_times:
      #print("spawn_loop")
      spawn_point_choice = 1000
      spawn_point_num = 0
      for point_num in range(0,len(self.vehicle_spawn_points)):
        #print("point_num_x : ",self.vehicle_spawn_points[point_num].location.x)
        #print("point_num_y : ",self.vehicle_spawn_points[point_num].location.y)
        spawn_math = math.sqrt((fin_x - self.vehicle_spawn_points[point_num].location.x)**2+(fin_y - self.vehicle_spawn_points[point_num].location.y)**2)
        if spawn_math < spawn_point_choice:
          spawn_point_choice = spawn_math
          spawn_point_num = point_num
      spawn_point = self.vehicle_spawn_points[spawn_point_num]
      if self._try_spawn_ego_vehicle_at(spawn_point):
        #print("DEBUG: Spawn Point is ", end="")
        #print(spawn_point.location)
        break
      else:
        if self.task_mode == 'source_is_dest' or '_division' in initial_task_mode:
          if len(current_quadrant_points) == 0:
            ego_spawn_times += self.max_ego_spawn_times + 1
          else:
            current_quadrant_points.remove(spawn_point)
        elif self.task_mode == 'sequential_point':
          dest_terminate = True
          terminal_waypoint = None
          mode_change_interval = 1 #Update route
          ego_spawn_times += self.max_ego_spawn_times + 1
        else:
          ego_spawn_times += 1
        time.sleep(0.1)
    print("Finish_point : ",self.finish_point)
    print("Spawn_Point : ",spawn_point)

    
    
    
    
    
    
    
    
    
    
    #New Spawn Point(ランダム)s
    '''
    while ego_spawn_times <= self.max_ego_spawn_times:
      #print("spawn_loop")
      spawn_point = random.choice(self.vehicle_spawn_points)
      if self._try_spawn_ego_vehicle_at(spawn_point):
        #print("DEBUG: Spawn Point is ", end="")
        #print(spawn_point.location)
        break
      else:
        if self.task_mode == 'source_is_dest' or '_division' in initial_task_mode:
          if len(current_quadrant_points) == 0:
            ego_spawn_times += self.max_ego_spawn_times + 1
          else:
            current_quadrant_points.remove(spawn_point)
        elif self.task_mode == 'sequential_point':
          dest_terminate = True
          terminal_waypoint = None
          mode_change_interval = 1 #Update route
          ego_spawn_times += self.max_ego_spawn_times + 1
        else:
          ego_spawn_times += 1
        time.sleep(0.1)
    print("Spawn_Point : ",spawn_point)
    '''
    
    
    #This episode is expired
    if ego_spawn_times > self.max_ego_spawn_times:
        print("DEBUG: max_spawn_loop")
        self.reset_loop_flag+=1
        self.reset()
    if self.reset_loop_flag == 1:
        self.reset_loop_flag-=1
        print("reset_loop")
        return
        
    # Destination
    """
    if initial_task_mode == 'simple_route_quad1_to_quad2':
      self.task_mode = 'second_quadrant'
    elif initial_task_mode == 'simple_route_quad2_to_quad3':
      self.task_mode = 'third_quadrant'
    elif initial_task_mode == 'simple_route_quad3_to_quad4':
      self.task_mode = 'fourth_quadrant'
    elif initial_task_mode == 'simple_route_quad4_to_quad1':
      self.task_mode = 'first_quadrant'
    elif 'general_route' in initial_task_mode or 'sequential_route' in initial_task_mode:
      self.task_mode = route[1]
      if 'sequential_point' in initial_task_mode:
        if dest_terminate == False and dest_point != None:
          self.task_mode = 'set_previous_dest'
    
    if self.task_mode == 'roundabout':
      self.dests = [[4.46, -61.46, 0], [-49.53, -2.89, 0], [-6.48, 55.47, 0], [35.96, 3.33, 0]]
    elif self.task_mode == 'first_quadrant':
      dest_point = random.choice(first_quadrant_points)
      while dest_point == spawn_point:
        dest_point = random.choice(first_quadrant_points)
      if '1_point_by_1_division' in initial_task_mode:
        dest_point = get_closest_vehicle_spawn_point(first_quadrant_points, [105,63])
      elif '4_points_by_1_division' in initial_task_mode:
        dest_points = []
        candidate = get_closest_vehicle_spawn_point(first_quadrant_points, [105,63])
        dest_points.append(candidate)
        candidate = get_closest_vehicle_spawn_point(first_quadrant_points, [201,63])
        dest_points.append(candidate)
        candidate = get_closest_vehicle_spawn_point(first_quadrant_points, [105,205])
        dest_points.append(candidate)
        candidate = get_closest_vehicle_spawn_point(first_quadrant_points, [242,133])
        dest_points.append(candidate)
        dest_point = random.choice(dest_points)
      elif '2_points_by_1_division' in initial_task_mode:
        dest_points = []
        candidate = get_closest_vehicle_spawn_point(first_quadrant_points, [105,63])
        dest_points.append(candidate)
        candidate = get_closest_vehicle_spawn_point(first_quadrant_points, [242,133])
        dest_points.append(candidate)
        dest_point = random.choice(dest_points)
      elif '8_points_by_1_division' in initial_task_mode:
        dest_points = []
        candidate = get_closest_vehicle_spawn_point(first_quadrant_points, [66,4])
        dest_points.append(candidate)
        candidate = get_closest_vehicle_spawn_point(first_quadrant_points, [137,4])
        dest_points.append(candidate)
        candidate = get_closest_vehicle_spawn_point(first_quadrant_points, [209,4])
        dest_points.append(candidate)
        candidate = get_closest_vehicle_spawn_point(first_quadrant_points, [105,63])
        dest_points.append(candidate)
        candidate = get_closest_vehicle_spawn_point(first_quadrant_points, [105,134])
        dest_points.append(candidate)
        candidate = get_closest_vehicle_spawn_point(first_quadrant_points, [66,205])
        dest_points.append(candidate)
        candidate = get_closest_vehicle_spawn_point(first_quadrant_points, [137,205])
        dest_points.append(candidate)
        candidate = get_closest_vehicle_spawn_point(first_quadrant_points, [209,195])
        dest_points.append(candidate)
        dest_point = random.choice(dest_points)
      self.dests = [[dest_point.location.x, dest_point.location.y, 0]]
    elif self.task_mode == 'second_quadrant':
      dest_point = random.choice(second_quadrant_points)
      while dest_point == spawn_point:
        dest_point = random.choice(second_quadrant_points)
      if '1_point_by_1_division' in initial_task_mode:
        dest_point = get_closest_vehicle_spawn_point(second_quadrant_points, [-84,63])
      elif '4_points_by_1_division' in initial_task_mode:
        dest_points = []
        candidate = get_closest_vehicle_spawn_point(second_quadrant_points, [-146,53])
        dest_points.append(candidate)
        candidate = get_closest_vehicle_spawn_point(second_quadrant_points, [-84,63])
        dest_points.append(candidate)
        candidate = get_closest_vehicle_spawn_point(second_quadrant_points, [-6,171])
        dest_points.append(candidate)
        candidate = get_closest_vehicle_spawn_point(second_quadrant_points, [-84,165])
        dest_points.append(candidate)
        dest_point = random.choice(dest_points)
      elif '2_points_by_1_division' in initial_task_mode:
        dest_points = []
        candidate = get_closest_vehicle_spawn_point(second_quadrant_points, [-146,53])
        dest_points.append(candidate)
        candidate = get_closest_vehicle_spawn_point(second_quadrant_points, [-6,171])
        dest_points.append(candidate)
        dest_point = random.choice(dest_points)
      elif '8_points_by_1_division' in initial_task_mode:
        dest_points = []
        candidate = get_closest_vehicle_spawn_point(second_quadrant_points, [-146,53])
        dest_points.append(candidate)
        candidate = get_closest_vehicle_spawn_point(second_quadrant_points, [-75,53])
        dest_points.append(candidate)
        candidate = get_closest_vehicle_spawn_point(second_quadrant_points, [-3,53])
        dest_points.append(candidate)
        candidate = get_closest_vehicle_spawn_point(second_quadrant_points, [-146,124])
        dest_points.append(candidate)
        candidate = get_closest_vehicle_spawn_point(second_quadrant_points, [-75,124])
        dest_points.append(candidate)
        candidate = get_closest_vehicle_spawn_point(second_quadrant_points, [-3,124])
        dest_points.append(candidate)
        candidate = get_closest_vehicle_spawn_point(second_quadrant_points, [-75,195])
        dest_points.append(candidate)
        candidate = get_closest_vehicle_spawn_point(second_quadrant_points, [-3,195])
        dest_points.append(candidate)
        dest_point = random.choice(dest_points)
      self.dests = [[dest_point.location.x, dest_point.location.y, 0]]
    elif self.task_mode == 'third_quadrant':
      dest_point = random.choice(third_quadrant_points)
      while dest_point == spawn_point:
        dest_point = random.choice(third_quadrant_points)
      if '1_point_by_1_division' in initial_task_mode:
        dest_point = get_closest_vehicle_spawn_point(third_quadrant_points, [-84,-123])
      elif '4_points_by_1_division' in initial_task_mode:
        dest_points = []
        candidate = get_closest_vehicle_spawn_point(third_quadrant_points, [-146,-90])
        dest_points.append(candidate)
        candidate = get_closest_vehicle_spawn_point(third_quadrant_points, [-84,-123])
        dest_points.append(candidate)
        candidate = get_closest_vehicle_spawn_point(third_quadrant_points, [-84,-52])
        dest_points.append(candidate)
        candidate = get_closest_vehicle_spawn_point(third_quadrant_points, [-17,-140])
        dest_points.append(candidate)
        dest_point = random.choice(dest_points)
      elif '2_points_by_1_division' in initial_task_mode:
        dest_points = []
        candidate = get_closest_vehicle_spawn_point(third_quadrant_points, [-146,-90])
        dest_points.append(candidate)
        candidate = get_closest_vehicle_spawn_point(third_quadrant_points, [-17,-140])
        dest_points.append(candidate)
        dest_point = random.choice(dest_points)
      elif '8_points_by_1_division' in initial_task_mode:
        dest_points = []
        candidate = get_closest_vehicle_spawn_point(third_quadrant_points, [-161,-10])
        dest_points.append(candidate)
        candidate = get_closest_vehicle_spawn_point(third_quadrant_points, [-161,-82])
        dest_points.append(candidate)
        candidate = get_closest_vehicle_spawn_point(third_quadrant_points, [-89,-10])
        dest_points.append(candidate)
        candidate = get_closest_vehicle_spawn_point(third_quadrant_points, [-89,-82])
        dest_points.append(candidate)
        candidate = get_closest_vehicle_spawn_point(third_quadrant_points, [-89,-153])
        dest_points.append(candidate)
        candidate = get_closest_vehicle_spawn_point(third_quadrant_points, [-18,-36])
        dest_points.append(candidate)
        candidate = get_closest_vehicle_spawn_point(third_quadrant_points, [-18,-107])
        dest_points.append(candidate)
        candidate = get_closest_vehicle_spawn_point(third_quadrant_points, [-18,-179])
        dest_points.append(candidate)
        dest_point = random.choice(dest_points)
      self.dests = [[dest_point.location.x, dest_point.location.y, 0]]
    elif self.task_mode == 'fourth_quadrant':
      dest_point = random.choice(fourth_quadrant_points)
      while dest_point == spawn_point:
        dest_point = random.choice(fourth_quadrant_points)
      if '1_point_by_1_division' in initial_task_mode:
        dest_point = get_closest_vehicle_spawn_point(fourth_quadrant_points, [76,-123])
      elif '4_points_by_1_division' in initial_task_mode:
        dest_points = []
        candidate = get_closest_vehicle_spawn_point(fourth_quadrant_points, [72,-201])
        dest_points.append(candidate)
        candidate = get_closest_vehicle_spawn_point(fourth_quadrant_points, [178,-201])
        dest_points.append(candidate)
        candidate = get_closest_vehicle_spawn_point(fourth_quadrant_points, [242,-10])
        dest_points.append(candidate)
        candidate = get_closest_vehicle_spawn_point(fourth_quadrant_points, [68,-10])
        dest_points.append(candidate)
        dest_point = random.choice(dest_points)
      elif '2_points_by_1_division' in initial_task_mode:
        dest_points = []
        candidate = get_closest_vehicle_spawn_point(fourth_quadrant_points, [72,-201])
        dest_points.append(candidate)
        candidate = get_closest_vehicle_spawn_point(fourth_quadrant_points, [242,-10])
        dest_points.append(candidate)
        dest_point = random.choice(dest_points)
      elif '8_points_by_1_division' in initial_task_mode:
        dest_points = []
        candidate = get_closest_vehicle_spawn_point(fourth_quadrant_points, [82,-201])
        dest_points.append(candidate)
        candidate = get_closest_vehicle_spawn_point(fourth_quadrant_points, [82,-130])
        dest_points.append(candidate)
        candidate = get_closest_vehicle_spawn_point(fourth_quadrant_points, [82,-58])
        dest_points.append(candidate)
        candidate = get_closest_vehicle_spawn_point(fourth_quadrant_points, [143,-201])
        dest_points.append(candidate)
        candidate = get_closest_vehicle_spawn_point(fourth_quadrant_points, [143,-130])
        dest_points.append(candidate)
        candidate = get_closest_vehicle_spawn_point(fourth_quadrant_points, [143,-58])
        dest_points.append(candidate)
        candidate = get_closest_vehicle_spawn_point(fourth_quadrant_points, [242,-117])
        dest_points.append(candidate)
        candidate = get_closest_vehicle_spawn_point(fourth_quadrant_points, [242,-10])
        dest_points.append(candidate)
        dest_point = random.choice(dest_points)
      self.dests = [[dest_point.location.x, dest_point.location.y, 0]]
    elif self.task_mode == 'set_previous_dest':
      self.dests = [[dest_point.location.x, dest_point.location.y, 0]]
    else:
      self.dests = None
    #print("DEBUG: Dest Point is ", end="")
    #print(dest_point.location)
    """
    # New Destination
    destination_count = 0
    dest_point = random.choice(self.vehicle_spawn_points)
    self.dests = [[dest_point.location.x, dest_point.location.y, 0]]
    while dest_point == spawn_point and destination_count <100:
      #print("dest_loop")
      destination_count+=1
      dest_point = random.choice(self.vehicle_spawn_points)
      self.dests = [[dest_point.location.x, dest_point.location.y, 0]]
    print("Destination :",self.dests)

    # Add collision sensor
    self.collision_sensor = self.world.spawn_actor(self.collision_bp, carla.Transform(), attach_to=self.ego)
    self.collision_sensor.listen(lambda event: get_collision_hist(event))
    def get_collision_hist(event):
      impulse = event.normal_impulse
      intensity = np.sqrt(impulse.x**2 + impulse.y**2 + impulse.z**2)
      self.collision_hist.append(intensity)
      if len(self.collision_hist)>self.collision_hist_l:
        self.collision_hist.pop(0)
    self.collision_hist = []

    # Add lidar sensor
    self.lidar_sensor = self.world.spawn_actor(self.lidar_bp, self.lidar_trans, attach_to=self.ego)
    self.lidar_sensor.listen(lambda data: get_lidar_data(data))
    def get_lidar_data(data):
      self.lidar_data = data

    # Add camera sensor
    self.camera_sensor = self.world.spawn_actor(self.camera_bp, self.camera_trans, attach_to=self.ego)
    self.camera_sensor.listen(lambda data: get_camera_img(data))
    def get_camera_img(data):
      array = np.frombuffer(data.raw_data, dtype = np.dtype("uint8"))
      array = np.reshape(array, (data.height, data.width, 4))
      array = array[:, :, :3]
      array = array[:, :, ::-1]
      self.camera_img = array

    # Update timesteps
    self.time_step=0
    self.reset_step+=1
    print(self.reset_step,'回目')
    print('time:', time.time())
    if self.reset_step == episode_sum + 1: #終了回数
      sys.exit()

    # Enable sync mode
    self.settings.synchronous_mode = True
    self.world.apply_settings(self.settings)

    self.routeplanner = RoutePlanner(self.ego, self.max_waypt)
    self.waypoints, _, self.vehicle_front = self.routeplanner.run_step()

    # Set ego information for render
    self.birdeye_render.set_hero(self.ego, self.ego.id)
    
    print("Reset_Finish")

    return self._get_obs()
  
  def step(self, action):
    #現在地の取得（終了時の座標を記録し、次のスポーン地点を終了地点付近に設定するため）
    self.finish_point = self.ego.get_location()
    # Calculate acceleration and steering
    if self.discrete:
      acc = self.discrete_act[0][action//self.n_steer]
      steer = self.discrete_act[1][action%self.n_steer]
    else:
      acc = action[0]
      steer = action[1]

    # Convert acceleration to throttle and brake
    if acc > 0:
      throttle = np.clip(acc/3,0,1)
      brake = 0
    else:
      throttle = 0
      brake = np.clip(-acc/8,0,1)

    # Apply control
    act = carla.VehicleControl(throttle=float(throttle), steer=float(-steer), brake=float(brake))
    self.ego.apply_control(act)

    self.world.tick()

    # Append actors polygon list
    vehicle_poly_dict = self._get_actor_polygons('vehicle.*')
    self.vehicle_polygons.append(vehicle_poly_dict)
    while len(self.vehicle_polygons) > self.max_past_step:
      self.vehicle_polygons.pop(0)
    walker_poly_dict = self._get_actor_polygons('walker.*')
    self.walker_polygons.append(walker_poly_dict)
    while len(self.walker_polygons) > self.max_past_step:
      self.walker_polygons.pop(0)

    # route planner
    self.waypoints, _, self.vehicle_front = self.routeplanner.run_step()

    # state information
    info = {
      'waypoints': self.waypoints,
      'vehicle_front': self.vehicle_front
    }
    
    # Update timesteps
    self.time_step += 1 #毎回rewardとかを算出
    self.total_step += 1

    return (self._get_obs(), self._get_reward(), self._terminal(), copy.deepcopy(info))

  def seed(self, seed=None):
    self.np_random, seed = seeding.np_random(seed)
    return [seed]

  def render(self, mode):
    pass

  def _create_vehicle_bluepprint(self, actor_filter, color=None, number_of_wheels=[4]):
    """Create the blueprint for a specific actor type.

    Args:
      actor_filter: a string indicating the actor type, e.g, 'vehicle.lincoln*'.

    Returns:
      bp: the blueprint object of carla.
    """
    blueprints = self.world.get_blueprint_library().filter(actor_filter)
    blueprint_library = []
    for nw in number_of_wheels:
      blueprint_library = blueprint_library + [x for x in blueprints if int(x.get_attribute('number_of_wheels')) == nw]
    bp = random.choice(blueprint_library)
    if bp.has_attribute('color'):
      if not color:
        color = random.choice(bp.get_attribute('color').recommended_values)
      bp.set_attribute('color', color)
    return bp

  def _init_renderer(self):
    """Initialize the birdeye view renderer.
    """
    pygame.init()
    self.display = pygame.display.set_mode(
    (self.display_size * 3, self.display_size),
    pygame.HWSURFACE | pygame.DOUBLEBUF)

    pixels_per_meter = self.display_size / self.obs_range
    pixels_ahead_vehicle = (self.obs_range/2 - self.d_behind) * pixels_per_meter
    birdeye_params = {
      'screen_size': [self.display_size, self.display_size],
      'pixels_per_meter': pixels_per_meter,
      'pixels_ahead_vehicle': pixels_ahead_vehicle
    }
    self.birdeye_render = BirdeyeRender(self.world, birdeye_params)

  def _set_synchronous_mode(self, synchronous = True):
    """Set whether to use the synchronous mode.
    """
    self.settings.synchronous_mode = synchronous
    self.world.apply_settings(self.settings)

  def _try_spawn_random_vehicle_at(self, transform, number_of_wheels=[4]):
    """Try to spawn a surrounding vehicle at specific transform with random bluprint.

    Args:
      transform: the carla transform object.

    Returns:
      Bool indicating whether the spawn is successful.
    """
    blueprint = self._create_vehicle_bluepprint('vehicle.*', number_of_wheels=number_of_wheels)
    blueprint.set_attribute('role_name', 'autopilot')
    vehicle = self.world.try_spawn_actor(blueprint, transform)
    if vehicle is not None:
      vehicle.set_autopilot()
      return True
    return False

  def _try_spawn_random_walker_at(self, transform):
    """Try to spawn a walker at specific transform with random bluprint.

    Args:
      transform: the carla transform object.

    Returns:
      Bool indicating whether the spawn is successful.
    """
    walker_bp = random.choice(self.world.get_blueprint_library().filter('walker.*'))
    # set as not invencible
    if walker_bp.has_attribute('is_invincible'):
      walker_bp.set_attribute('is_invincible', 'false')
    walker_actor = self.world.try_spawn_actor(walker_bp, transform)

    if walker_actor is not None:
      walker_controller_bp = self.world.get_blueprint_library().find('controller.ai.walker')
      walker_controller_actor = self.world.spawn_actor(walker_controller_bp, carla.Transform(), walker_actor)
      # start walker
      walker_controller_actor.start()
      # set walk to random point
      walker_controller_actor.go_to_location(self.world.get_random_location_from_navigation())
      # random max speed
      walker_controller_actor.set_max_speed(1 + random.random())    # max speed between 1 and 2 (default is 1.4 m/s)
      return True
    return False

  def _try_spawn_ego_vehicle_at(self, transform):
    """Try to spawn the ego vehicle at specific transform.
    Args:
      transform: the carla transform object.
    Returns:
      Bool indicating whether the spawn is successful.
    """
    vehicle = None
    # Check if ego position overlaps with surrounding vehicles
    overlap = False
    for idx, poly in self.vehicle_polygons[-1].items():
      poly_center = np.mean(poly, axis=0)
      ego_center = np.array([transform.location.x, transform.location.y])
      dis = np.linalg.norm(poly_center - ego_center)
      if dis > 8:
        continue
      else:
        overlap = True
        break

    if not overlap:
      vehicle = self.world.try_spawn_actor(self.ego_bp, transform)

    if vehicle is not None:
      self.ego=vehicle
      return True
      
    return False

  def _get_actor_polygons(self, filt):
    """Get the bounding box polygon of actors.

    Args:
      filt: the filter indicating what type of actors we'll look at.

    Returns:
      actor_poly_dict: a dictionary containing the bounding boxes of specific actors.
    """
    actor_poly_dict={}
    for actor in self.world.get_actors().filter(filt):
      # Get x, y and yaw of the actor
      trans=actor.get_transform()
      x=trans.location.x
      y=trans.location.y
      yaw=trans.rotation.yaw/180*np.pi
      # Get length and width
      bb=actor.bounding_box
      l=bb.extent.x
      w=bb.extent.y
      # Get bounding box polygon in the actor's local coordinate
      poly_local=np.array([[l,w],[l,-w],[-l,-w],[-l,w]]).transpose()
      # Get rotation matrix to transform to global coordinate
      R=np.array([[np.cos(yaw),-np.sin(yaw)],[np.sin(yaw),np.cos(yaw)]])
      # Get global bounding box polygon
      poly=np.matmul(R,poly_local).transpose()+np.repeat([[x,y]],4,axis=0)
      actor_poly_dict[actor.id]=poly
    return actor_poly_dict

  def _get_obs(self):
    """Get the observations."""
    ## Birdeye rendering
    self.birdeye_render.vehicle_polygons = self.vehicle_polygons
    self.birdeye_render.walker_polygons = self.walker_polygons
    self.birdeye_render.waypoints = self.waypoints

    # birdeye view with roadmap and actors
    birdeye_render_types = ['roadmap', 'actors']
    if self.display_route:
      birdeye_render_types.append('waypoints')
    self.birdeye_render.render(self.display, birdeye_render_types)
    birdeye = pygame.surfarray.array3d(self.display)
    birdeye = birdeye[0:self.display_size, :, :]
    birdeye = display_to_rgb(birdeye, self.obs_size)

    # Roadmap
    if self.pixor:
      roadmap_render_types = ['roadmap']
      if self.display_route:
        roadmap_render_types.append('waypoints')
      self.birdeye_render.render(self.display, roadmap_render_types)
      roadmap = pygame.surfarray.array3d(self.display)
      roadmap = roadmap[0:self.display_size, :, :]
      roadmap = display_to_rgb(roadmap, self.obs_size)
      # Add ego vehicle
      for i in range(self.obs_size):
        for j in range(self.obs_size):
          if abs(birdeye[i, j, 0] - 255)<20 and abs(birdeye[i, j, 1] - 0)<20 and abs(birdeye[i, j, 0] - 255)<20:
            roadmap[i, j, :] = birdeye[i, j, :]

    # Display birdeye image
    birdeye_surface = rgb_to_display_surface(birdeye, self.display_size)
    self.display.blit(birdeye_surface, (0, 0))

    ## Lidar image generation
    point_cloud = []
    # Get point cloud data
    for location in self.lidar_data:
      point_cloud.append([location.x, location.y, -location.z])
    point_cloud = np.array(point_cloud)
    # Separate the 3D space to bins for point cloud, x and y is set according to self.lidar_bin,
    # and z is set to be two bins.
    y_bins = np.arange(-(self.obs_range - self.d_behind), self.d_behind+self.lidar_bin, self.lidar_bin)
    x_bins = np.arange(-self.obs_range/2, self.obs_range/2+self.lidar_bin, self.lidar_bin)
    z_bins = [-self.lidar_height-1, -self.lidar_height+0.25, 1]
    # Get lidar image according to the bins
    lidar, _ = np.histogramdd(point_cloud, bins=(x_bins, y_bins, z_bins))
    lidar[:,:,0] = np.array(lidar[:,:,0]>0, dtype=np.uint8)
    lidar[:,:,1] = np.array(lidar[:,:,1]>0, dtype=np.uint8)
    # Add the waypoints to lidar image
    if self.display_route:
      wayptimg = (birdeye[:,:,0] <= 10) * (birdeye[:,:,1] <= 10) * (birdeye[:,:,2] >= 240)
    else:
      wayptimg = birdeye[:,:,0] < 0  # Equal to a zero matrix
    wayptimg = np.expand_dims(wayptimg, axis=2)
    wayptimg = np.fliplr(np.rot90(wayptimg, 3))

    # Get the final lidar image
    lidar = np.concatenate((lidar, wayptimg), axis=2)
    lidar = np.flip(lidar, axis=1)
    lidar = np.rot90(lidar, 1)
    lidar = lidar * 255

    # Display lidar image
    lidar_surface = rgb_to_display_surface(lidar, self.display_size)
    self.display.blit(lidar_surface, (self.display_size, 0))

    ## Display camera image
    camera = resize(self.camera_img, (self.obs_size, self.obs_size)) * 255
    camera_surface = rgb_to_display_surface(camera, self.display_size)
    self.display.blit(camera_surface, (self.display_size * 2, 0))

    # Display on pygame
    pygame.display.flip()

    # State observation
    ego_trans = self.ego.get_transform()
    ego_x = ego_trans.location.x
    ego_y = ego_trans.location.y
    ego_yaw = ego_trans.rotation.yaw/180*np.pi
    lateral_dis, w = get_preview_lane_dis(self.waypoints, ego_x, ego_y)
    delta_yaw = np.arcsin(np.cross(w, 
      np.array(np.array([np.cos(ego_yaw), np.sin(ego_yaw)]))))
    v = self.ego.get_velocity()
    speed = np.sqrt(v.x**2 + v.y**2)
    state = np.array([lateral_dis, - delta_yaw, speed, self.vehicle_front])

    if self.pixor:
      ## Vehicle classification and regression maps (requires further normalization)
      vh_clas = np.zeros((self.pixor_size, self.pixor_size))
      vh_regr = np.zeros((self.pixor_size, self.pixor_size, 6))

      # Generate the PIXOR image. Note in CARLA it is using left-hand coordinate
      # Get the 6-dim geom parametrization in PIXOR, here we use pixel coordinate
      for actor in self.world.get_actors().filter('vehicle.*'):
        x, y, yaw, l, w = get_info(actor)
        x_local, y_local, yaw_local = get_local_pose((x, y, yaw), (ego_x, ego_y, ego_yaw))
        if actor.id != self.ego.id:
          if abs(y_local)<self.obs_range/2+1 and x_local<self.obs_range-self.d_behind+1 and x_local>-self.d_behind-1:
            x_pixel, y_pixel, yaw_pixel, l_pixel, w_pixel = get_pixel_info(
              local_info=(x_local, y_local, yaw_local, l, w),
              d_behind=self.d_behind, obs_range=self.obs_range, image_size=self.pixor_size)
            cos_t = np.cos(yaw_pixel)
            sin_t = np.sin(yaw_pixel)
            logw = np.log(w_pixel)
            logl = np.log(l_pixel)
            pixels = get_pixels_inside_vehicle(
              pixel_info=(x_pixel, y_pixel, yaw_pixel, l_pixel, w_pixel),
              pixel_grid=self.pixel_grid)
            for pixel in pixels:
              vh_clas[pixel[0], pixel[1]] = 1
              dx = x_pixel - pixel[0]
              dy = y_pixel - pixel[1]
              vh_regr[pixel[0], pixel[1], :] = np.array(
                [cos_t, sin_t, dx, dy, logw, logl])

      # Flip the image matrix so that the origin is at the left-bottom
      vh_clas = np.flip(vh_clas, axis=0)
      vh_regr = np.flip(vh_regr, axis=0)

      # Pixor state, [x, y, cos(yaw), sin(yaw), speed]
      pixor_state = [ego_x, ego_y, np.cos(ego_yaw), np.sin(ego_yaw), speed]

    obs = {
      'camera':camera.astype(np.uint8),
      'lidar':lidar.astype(np.uint8),
      'birdeye':birdeye.astype(np.uint8),
      'state': state,
    }

    if self.pixor:
      obs.update({
        'roadmap':roadmap.astype(np.uint8),
        'vh_clas':np.expand_dims(vh_clas, -1).astype(np.float32),
        'vh_regr':vh_regr.astype(np.float32),
        'pixor_state': pixor_state,
      })

    return obs

  def _get_reward(self):
    """Calculate the step reward."""
    # reward for speed tracking
    v = self.ego.get_velocity()
    speed = np.sqrt(v.x**2 + v.y**2)
    r_speed = -abs(speed - self.desired_speed)
    
    # reward for collision
    r_collision = 0
    if len(self.collision_hist) > 0:
      r_collision = -1

    # reward for steering:
    r_steer = -self.ego.get_control().steer**2

    # reward for out of lane
    ego_x, ego_y = get_pos(self.ego)
    dis, w, waypt = get_lane_dis(self.waypoints, ego_x, ego_y)
    r_out = 0
    if abs(dis) > self.out_lane_thres:
      r_out = -1

    # longitudinal speed
    lspeed = np.array([v.x, v.y])
    lspeed_lon = np.dot(lspeed, w)

    # cost for too fast
    r_fast = 0
    if lspeed_lon > self.desired_speed:
      r_fast = -1

    # cost for lateral acceleration
    r_lat = - abs(self.ego.get_control().steer) * lspeed_lon**2

    r = 200*r_collision + 1*lspeed_lon + 10*r_fast + 1*r_out + r_steer*5 + 0.2*r_lat - 0.1

    print('r_lat :\t\t',r_lat)
    print('r_steer :\t',r_steer)
    print('r_out :\t\t',r_out)
    print('r_fast :\t',r_fast)
    print('lspeed_lon :\t',lspeed_lon)
    print('r_collision :\t',r_collision)
    print('reward :\t',r)
    print('spawn_num :\t',len(self.vehicle_spawn_points))
    return r
    

  def _terminal(self):
    global initial_task_mode
    global mode_change_interval
    global terminal_waypoint
    global dest_point
    global spawn_point
    global col_terminate
    global max_step_terminate
    global dest_terminate
    global out_of_lane_terminate
    """Calculate whether to terminate the current episode."""
    # Get ego state
    ego_x, ego_y = get_pos(self.ego)

    ego_trans = self.ego.get_transform()
    if 'sequential_point' in initial_task_mode:
      terminal_waypoint = self.world.get_map().get_waypoint(ego_trans.location, project_to_road=True, lane_type=carla.LaneType.Driving)
      if terminal_waypoint == None:
        mode_change_interval = 1 #Update route
      else:
        mode_change_interval = self.reset_step + 1 #Do not update route
    
    col_terminate = False
    max_step_terminate = False
    dest_terminate = False
    out_of_lane_terminate = False

    # If collides
    if len(self.collision_hist)>0:
      print("DEBUG: col_terminate")
      col_terminate = True
      return True

    # If reach maximum timestep
    if self.time_step>self.max_time_episode:
      print("DEBUG: max_step_terminate")
      max_step_terminate = True
      return True

    # If at destination
    if self.dests is not None: # If at destination
      for dest in self.dests:
        if np.sqrt((ego_x-dest[0])**2+(ego_y-dest[1])**2)<4:
          print("DEBUG: dest_terminate")
          dest_terminate = True
          if 'sequential_point' in initial_task_mode:
            mode_change_interval = 1 #Update route
          return True

    # If out of lane
    dis, _, waypt = get_lane_dis(self.waypoints, ego_x, ego_y)
    if abs(dis) > self.out_lane_thres:
      print("DEBUG: out_of_lane_terminate")
      out_of_lane_terminate = True
      return True

    return False

  def _clear_all_actors(self, actor_filters):
    """Clear specific actors."""
    for actor_filter in actor_filters:
      for actor in self.world.get_actors().filter(actor_filter):
        if actor.is_alive:
          if actor.type_id == 'controller.ai.walker':
            actor.stop()
          actor.destroy()
