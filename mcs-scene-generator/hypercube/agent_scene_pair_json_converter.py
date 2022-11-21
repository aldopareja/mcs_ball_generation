import copy
import logging
import math
import random
import uuid
from typing import Any, Callable, Dict, List, Optional, Tuple

from shapely import ops

from generator import (
    MaterialTuple,
    ObjectBounds,
    Scene,
    SceneException,
    geometry,
    materials,
    structures,
    tags
)
from generator.separating_axis_theorem import sat_entry

from .hypercubes import update_scene_objects

logger = logging.getLogger(__name__)


class ObjectConfig():
    def __init__(
        self,
        object_type: str,
        scale_xz: float,
        scale_y: float,
        rotation_x: float = 0,
        rotation_y: float = 0,
        rotation_z: float = 0,
        untrained: bool = False,
    ) -> None:
        self.object_type = object_type
        self.scale_xz = scale_xz
        self.scale_y = scale_y
        self.rotation_x = rotation_x
        self.rotation_y = rotation_y
        self.rotation_z = rotation_z
        self.untrained = untrained


class AgentConfig(ObjectConfig):
    def __init__(
        self,
        object_type: str,
        scale: float,
        rotation_x: float = 0,
        rotation_y: float = 0,
        rotation_z: float = 0,
        untrained: bool = False
    ) -> None:
        super().__init__(
            object_type,
            scale,
            scale,
            rotation_x,
            rotation_y,
            rotation_z,
            untrained
        )


class ObjectDimensions():
    def __init__(
        self,
        object_type: str,
        x: float,
        y: float,
        z: float,
        center_y: float = None
    ) -> None:
        self.object_type = object_type
        self.x = x
        self.y = y
        self.z = z
        self.center_y = (y / 2.0) if center_y is None else center_y


class ObjectConfigWithMaterial(ObjectConfig):
    def __init__(
        self,
        config: ObjectConfig,
        material: Tuple[str, List[str]]
    ) -> None:
        super().__init__(
            config.object_type,
            config.scale_xz,
            config.scale_y,
            config.rotation_x,
            config.rotation_y,
            config.rotation_z,
            config.untrained
        )
        self.material = material


# Debug logging
SAVE_TRIALS_TO_FILE = False
TRIALS_SUFFIX = '_trials.txt'

EXPECTED = 'expected'
UNEXPECTED = 'unexpected'

# Use a 10x10 grid, with each cell 0.5 x 0.5, including the border wall.
GRID_MIN_X = -2.5
GRID_MAX_X = 2.5
GRID_MIN_Z = -2.5
GRID_MAX_Z = 2.5

JSON_BORDER_WALL_MIN_X = 0
JSON_BORDER_WALL_MIN_Z = 0
JSON_BORDER_WALL_MAX_X = 180
JSON_BORDER_WALL_MAX_Z = 180

# The wait time in steps before and after the agent's movement in each trial.
STARTING_STEP_WAIT_TIME = 3
PAUSED_STEP_WAIT_TIME = 2
DEFUSE_STEP_SKIP_TIME = 5
POST_DEFUSE_WAIT_TIME = 5

OBJECT_DIMENSIONS = {
    'blob_01': ObjectDimensions('blob_01', 0.26, 0.8, 0.36),
    'blob_02': ObjectDimensions('blob_02', 0.33, 0.78, 0.33),
    'blob_03': ObjectDimensions('blob_03', 0.25, 0.69, 0.25),
    'blob_04': ObjectDimensions('blob_04', 0.3, 0.53, 0.3, 0.225),
    'blob_05': ObjectDimensions('blob_05', 0.38, 0.56, 0.38, 0.24),
    'blob_06': ObjectDimensions('blob_06', 0.52, 0.5, 0.54),
    'blob_07': ObjectDimensions('blob_07', 0.25, 0.55, 0.25, 0.245),
    'blob_08': ObjectDimensions('blob_08', 0.27, 0.62, 0.15),
    'blob_09': ObjectDimensions('blob_09', 0.33, 0.78, 0.44),
    'blob_10': ObjectDimensions('blob_10', 0.24, 0.5, 0.24),
    'circle_frustum': ObjectDimensions('circle_frustum', 1, 1, 1),
    'cone': ObjectDimensions('cone', 1, 1, 1),
    'cube': ObjectDimensions('cube', 1, 1, 1),
    'cube_hollow_narrow': ObjectDimensions('cube_hollow_narrow', 1, 1, 1, 0),
    'cube_hollow_wide': ObjectDimensions('cube_hollow_wide', 1, 1, 1, 0),
    'cylinder': ObjectDimensions('cylinder', 1, 2, 1),
    'hash': ObjectDimensions('hash', 1, 1, 1, 0),
    'letter_x': ObjectDimensions('letter_x', 1, 1, 1, 0),
    'lock_wall': ObjectDimensions('lock_wall', 1, 1, 1),
    'pyramid': ObjectDimensions('pyramid', 1, 1, 1),
    'semi_sphere': ObjectDimensions('semi_sphere', 1, 1, 1, 0.25),
    'sphere': ObjectDimensions('sphere', 1, 1, 1),
    'square_frustum': ObjectDimensions('square_frustum', 1, 1, 1),
    'triangle': ObjectDimensions('triangle', 1, 1, 1),
    'tube_narrow': ObjectDimensions('tube_narrow', 1, 1, 1),
    'tube_wide': ObjectDimensions('tube_wide', 1, 1, 1),
}

AGENT_OBJECT_CONFIG_LIST = [
    # Scaled to ensure that each agent's max X/Z dimension is 1
    AgentConfig('blob_01', 2.77),
    AgentConfig('blob_02', 3.03),
    AgentConfig('blob_03', 4),
    AgentConfig('blob_04', 3.33),
    AgentConfig('blob_05', 2.63),
    AgentConfig('blob_06', 1.85),
    AgentConfig('blob_07', 4),
    AgentConfig('blob_08', 3.7),
    AgentConfig('blob_09', 2.27),
    AgentConfig('blob_10', 4.16),
]
AGENT_OBJECT_MATERIAL_LIST = [
    materials.BLUE,
    materials.GOLDENROD,
    materials.GREEN,
    materials.PURPLE
    # Don't use red here because it looks too much like the maroon key.
]

GOAL_OBJECT_CONFIG_LIST = [
    # Scaled to ensure that each object's max X/Z dimension is 1
    ObjectConfig('circle_frustum', 1, 1),
    ObjectConfig('cube', 1, 1),
    ObjectConfig('cube_hollow_narrow', 1, 1),
    ObjectConfig('cube_hollow_wide', 1, 1),
    ObjectConfig('cylinder', 1, 0.5),
    ObjectConfig('letter_x', 1, 1),
    ObjectConfig('hash', 1, 1),
    ObjectConfig('pyramid', 1, 1),
    ObjectConfig('semi_sphere', 1, 1),
    ObjectConfig('sphere', 1, 1),
    ObjectConfig('square_frustum', 1, 1),
    ObjectConfig('tube_narrow', 1, 1),
    ObjectConfig('tube_wide', 1, 1),
]
GOAL_OBJECT_MATERIAL_LIST = [
    materials.AZURE,
    materials.BROWN,
    materials.CHARTREUSE,
    materials.CYAN,
    materials.GREY,
    materials.INDIGO,
    materials.NAVY,
    materials.OLIVE,
    materials.ORANGE,
    materials.ROSE,
    materials.SPRINGGREEN,
    materials.TEAL,
    materials.VIOLET,
    materials.YELLOW
]

# Make the home object as short as possible, without it looking weird in Unity.
HOME_OBJECT_HEIGHT = [0.01, 0.02]
HOME_OBJECT_MATERIAL = MaterialTuple('Custom/Materials/Magenta', ['magenta'])
HOME_OBJECT_SIZE = [0.5, 0.5]

WALL_OBJECT_HEIGHT = [0.0625, 0.125]
WALL_OBJECT_MATERIAL = MaterialTuple('Custom/Materials/Black', ['black'])
WALL_OBJECT_SIZE = [0.5, 0.5]

FUSE_WALL_OBJECT_HEIGHT = [0.05, 0.1]
FUSE_WALL_OBJECT_MATERIAL = MaterialTuple('Custom/Materials/Lime', ['lime'])
FUSE_WALL_OBJECT_SIZE = [0.495, 0.495]

KEY_OBJECT_HEIGHT = [FUSE_WALL_OBJECT_HEIGHT[0], 0.35]
KEY_OBJECT_MATERIAL = MaterialTuple('Custom/Materials/Maroon', ['maroon'])
KEY_OBJECT_SIZE = [FUSE_WALL_OBJECT_HEIGHT[1], 0.35]
KEY_OBJECT_TYPE = 'triangle'
KEY_OBJECT_ROTATION_X = 0
KEY_OBJECT_ROTATION_Y = {
    'positive_z': {
        'dimensions_x': 0.5,
        'dimensions_z': 0.25,
        'position_x': 0,
        'position_z': -0.25,
        'rotation_y': -45
    },
    'negative_z': {
        'dimensions_x': 0.5,
        'dimensions_z': 0.25,
        'position_x': 0,
        'position_z': 0.25,
        'rotation_y': 135
    },
    'positive_x': {
        'dimensions_x': 0.25,
        'dimensions_z': 0.5,
        'position_x': -0.25,
        'position_z': 0,
        'rotation_y': 45
    },
    'negative_x': {
        'dimensions_x': 0.25,
        'dimensions_z': 0.5,
        'position_x': 0.25,
        'position_z': 0,
        'rotation_y': -135
    }
}
KEY_OBJECT_ROTATION_Z = 90

LOCK_WALL_OBJECT_HEIGHT = FUSE_WALL_OBJECT_HEIGHT
LOCK_WALL_OBJECT_MATERIAL = FUSE_WALL_OBJECT_MATERIAL
LOCK_WALL_OBJECT_SIZE = FUSE_WALL_OBJECT_SIZE

# The floor and room walls should have bland colors and simple textures.
CEILING_MATERIAL = MaterialTuple("AI2-THOR/Materials/Walls/Drywall", ["white"])
FLOOR_OR_WALL_MATERIALS = [
    MaterialTuple("AI2-THOR/Materials/Ceramics/BrownMarbleFake 1", ["brown"]),
    MaterialTuple("AI2-THOR/Materials/Ceramics/ConcreteBoards1", ["grey"]),
    MaterialTuple("AI2-THOR/Materials/Ceramics/ConcreteFloor", ["grey"]),
    MaterialTuple("AI2-THOR/Materials/Ceramics/GREYGRANITE", ["grey"]),
    MaterialTuple("AI2-THOR/Materials/Ceramics/PinkConcrete_Bedroom1",
                  ["red"]),
    MaterialTuple("AI2-THOR/Materials/Ceramics/WhiteCountertop", ["grey"]),
    MaterialTuple("AI2-THOR/Materials/Wood/BedroomFloor1", ["brown"]),
    MaterialTuple("AI2-THOR/Materials/Wood/LightWoodCounters 1", ["brown"]),
    MaterialTuple("AI2-THOR/Materials/Wood/LightWoodCounters3", ["brown"]),
    MaterialTuple("AI2-THOR/Materials/Wood/LightWoodCounters4", ["brown"]),
    MaterialTuple(
        "AI2-THOR/Materials/Wood/TexturesCom_WoodFine0050_1_seamless_S",
        ["brown"]),
    MaterialTuple("AI2-THOR/Materials/Wood/WhiteWood", ["white"]),
    MaterialTuple("AI2-THOR/Materials/Wood/WoodFloorsCross", ["brown"]),
    MaterialTuple("AI2-THOR/Materials/Wood/WoodGrain_Brown", ["brown"]),
    MaterialTuple("AI2-THOR/Materials/Wood/WoodGrain_Tan", ["brown"])
]
FLOOR_MATERIALS = FLOOR_OR_WALL_MATERIALS + [
    MaterialTuple("AI2-THOR/Materials/Fabrics/Carpet2", ["brown"]),
    MaterialTuple("AI2-THOR/Materials/Fabrics/CarpetWhite", ["white"]),
    MaterialTuple("AI2-THOR/Materials/Fabrics/CarpetWhite 3", ["white"])
]
WALL_MATERIALS = FLOOR_OR_WALL_MATERIALS + [
    MaterialTuple("AI2-THOR/Materials/Walls/Drywall", ["white"]),
    MaterialTuple("AI2-THOR/Materials/Walls/DrywallBeige", ["white"]),
    MaterialTuple("AI2-THOR/Materials/Walls/Drywall4Tiled", ["white"]),
    MaterialTuple("AI2-THOR/Materials/Walls/WallDrywallGrey", ["grey"]),
    MaterialTuple("AI2-THOR/Materials/Walls/YellowDrywall", ["yellow"])
]


def _append_each_show_to_object(
    mcs_object: Dict[str, Any],
    trial: List[Dict[str, Any]],
    trial_start_step: int,
    json_property: str,
    unit_size: Tuple[float, float],
    on_step_callback: Callable = None,
    rotation_y: int = 0
) -> Dict[str, Any]:
    """Append a "shows" array element to the given moving object for each step
    in the given trial list."""

    # Add data for the object's movement across the frames to each step.
    step = trial_start_step
    for frame_index, frame in enumerate(trial):
        json_object = frame.get(json_property)
        # (Sometimes the key is not in some trials.)
        if json_object:
            json_object = (
                json_object if json_property == 'agent' else json_object[0]
            )
            json_coords = json_object[0]
            json_radius = json_object[1]
            json_size = [json_radius * 2, json_radius * 2]
            # Move the object to its new position for the step.
            mcs_show = _create_show(
                step,
                mcs_object['type'],
                mcs_object['debug']['configHeight'],
                mcs_object['debug']['configSize'],
                json_coords,
                json_size,
                unit_size,
                rotation_y
            )
            mcs_object['shows'].append(mcs_show)
            if on_step_callback:
                mcs_object = on_step_callback(
                    trial_start_step,
                    json_object,
                    mcs_object
                )
        step += 1
        mcs_object['debug']['boundsAtStep'].append(
            mcs_object['shows'][-1]['boundingBox']
        )

    # Add 1 for the EndHabituation action step at the end of the trial.
    step += 1
    mcs_object['debug']['boundsAtStep'].append(
        mcs_object['shows'][-1]['boundingBox']
    )

    # Remove the scale from each element in 'shows' except for the first, or
    # it will really mess up the simulation.
    for show in mcs_object['shows'][1:]:
        if 'scale' in show:
            del show['scale']

    return mcs_object


def _choose_config_list(
    trial_list: List[List[Dict[str, Any]]],
    config_list: List[Dict[str, Any]],
    object_type_list: List[str],
    material_list: List[str],
    json_property: str,
    used_type_list: List[str],
    used_material_list: List[str]
) -> List[ObjectConfigWithMaterial]:
    """Choose and return the shape and color of each object in the scene to use
    in both scenes across the pair so they always have the same config."""

    object_config_list = []

    # Retrieve the relevant data from the first frame of the first trial.
    # Assume the number of objects will never change across trials, and
    # objects will never change shape/color across trials/frames.
    object_count = len(trial_list[0][0][json_property])

    # Always return two agent configs, since doing so is trivial. We can just
    # ignore the 2nd if it's not needed.
    if json_property == 'agent':
        object_count = 2
        # Remove agent objects with pointy tops (e.g. cones, pyramids) from
        # scenes with keys (e.g. instrumental action) as requested by NYU.
        # Old prop: 'key' ... New prop: 'pin'
        if trial_list[0][0].get('pin', trial_list[0][0].get('key')):
            config_list = [config for config in config_list if (
                config.object_type != 'cone' and
                config.object_type != 'pyramid'
            )]

    # Randomly choose each object's shape and color config.
    for index in range(object_count):
        # Filter on type specified via command line argument.
        filtered_config_list = [
            config for config in config_list
            if config.object_type == object_type_list[index]
        ] if object_type_list[index] else config_list

        # Fall back on the original list.
        if not filtered_config_list:
            filtered_config_list = config_list

        # Filter out used types.
        filtered_config_list = [
            config for config in filtered_config_list
            if config.object_type not in used_type_list
        ]
        if not filtered_config_list:
            raise SceneException(
                f'No more available {json_property} types: {used_type_list=}'
            )

        # Filter out used materials.
        filtered_material_list = [
            material for material in material_list
            if material[0] not in used_material_list
        ]
        if not filtered_material_list:
            raise SceneException(
                f'No more available {json_property} materials: '
                f'{used_material_list=}'
            )

        chosen_config = random.choice(filtered_config_list)

        # Choose a random Y rotation for each agent.
        if json_property == 'agent' and chosen_config.rotation_y is None:
            chosen_config.rotation_y = random.choice([0, 90, 180, 270])

        # Choose a random object config (type/size) and material.
        config_with_material = ObjectConfigWithMaterial(
            chosen_config,
            random.choice(filtered_material_list)
        )
        object_config_list.append(config_with_material)

        # Add the chosen type and material to the "used" lists.
        used_type_list.append(config_with_material.object_type)
        used_material_list.append(config_with_material.material[0])

    return object_config_list


def _create_action_list(
    trial_list: List[List[Dict[str, Any]]]
) -> List[List[str]]:
    """Create and return the MCS scene's action list using the given trial
    list from the JSON file data."""
    action_list = []
    for index in range(0, len(trial_list)):
        # Add 1 for the EndHabituation action step at the end of the trial.
        total_steps = len(trial_list[index]) + 1
        logger.info(
            f'Trial={index+1} Frames={len(trial_list[index])} Steps='
            f'{total_steps}'
        )
        action_list.extend([['Pass']] * (total_steps - 1))
        action_list.append(['EndHabituation'])
    # Remove the EndHabituation action from the last test trial.
    return action_list[:-1]


def _create_agent_object_list(
    trial_list: List[List[Dict[str, Any]]],
    agent_object_config_list: List[ObjectConfigWithMaterial],
    unit_size: Tuple[float, float]
) -> List[Dict[str, Any]]:
    """Create and return the MCS scene's agent object list using the given
    trial list from the JSON file data."""

    agent_object_list = []
    mcs_agent_collection = {}

    # Retrieve the agent data for each trial from the trial's first frame.
    # Assume one agent per trial and it will never change shape/color.
    json_agent_collection = {}
    for trial in trial_list:
        json_agent = trial[0]['agent']
        json_icon = json_agent[2]
        if json_icon not in json_agent_collection:
            json_agent_collection[json_icon] = json_agent

    # Create each unique agent using its corresponding data from the JSON file.
    for index, json_icon in enumerate(list(json_agent_collection.keys())):
        json_agent = json_agent_collection[json_icon]
        json_coords = json_agent[0]
        json_radius = json_agent[1]
        json_size = [json_radius * 2, json_radius * 2]

        # Create the MCS agent object.
        config_with_material = agent_object_config_list[index]
        dimensions = OBJECT_DIMENSIONS[config_with_material.object_type]
        # Multiply the agent's scale based on its JSON radius and unit size.
        factor = json_radius * 2 * min(unit_size[0], unit_size[1])
        scale_xz = config_with_material.scale_xz * factor
        scale_y = config_with_material.scale_y * factor
        center_y = dimensions.center_y * config_with_material.scale_y * factor
        agent_object = _create_object(
            'agent_',
            config_with_material.object_type,
            config_with_material.material,
            [center_y, scale_y],
            [scale_xz, scale_xz],
            json_coords,
            json_size,
            unit_size,
            rotation_y=config_with_material.rotation_y
        )
        agent_object['hides'] = []
        # Set kinematic to avoid awkward shifting due to collision issues.
        agent_object['kinematic'] = True
        # Set physics so this agent's info is returned in the oracle metadata.
        agent_object['physics'] = True
        agent_object['debug'][
            tags.SCENE.UNTRAINED_SHAPE
        ] = config_with_material.untrained

        # Remove the object's first appearance (we will override it later).
        agent_object['shows'] = []
        agent_object['debug']['boundsAtStep'] = []

        # Save the agent in this function's output list.
        agent_object_list.append(agent_object)

        # Link the agent and its icon so we can update its movement across each
        # frame in the next code block.
        mcs_agent_collection[json_icon] = agent_object

    # Update each MCS agent object with its movement in each trials' frames.
    for trial_index, trial in enumerate(trial_list):
        trial_json_agent = trial[0]['agent']
        trial_json_icon = trial_json_agent[2]
        step = _identify_trial_index_starting_step(trial_index, trial_list)
        for json_icon, mcs_agent in mcs_agent_collection.items():
            # If the current agent is in this trial, then update its movement.
            if json_icon == trial_json_icon:
                _append_each_show_to_object(
                    mcs_agent,
                    trial,
                    step,
                    'agent',
                    unit_size,
                    rotation_y=mcs_agent['debug']['configRotation']
                )
            # Else, hide the current agent during this trial.
            else:
                mcs_agent['hides'].append({
                    'stepBegin': step
                })
                for _ in range(len(trial) + 1):
                    mcs_agent['debug']['boundsAtStep'].append(None)

    return agent_object_list


def _create_fuse_wall_object_list(
    trial_list: List[List[Dict[str, Any]]],
    unit_size: Tuple[float, float]
) -> List[Dict[str, Any]]:
    """Create and return the MCS scene's green fuse wall object list (used in
    the instrumental action scenes) using the given trial list from the JSON
    file data."""

    fuse_wall_object_list = []

    # Retrieve the complete fuse walls data from each frame of each trial.
    # Assume each trial will have new fuse walls, and all of the fuse walls in
    # a trial will be hidden on a specific frame.
    for trial_index, trial in enumerate(trial_list):
        # Identify the step on which this trial starts.
        step = _identify_trial_index_starting_step(trial_index, trial_list)

        # Must save object references to find removed walls.
        original_coords_to_object = {}

        # Generate the list of fuse walls used in this trial.
        for json_wall in trial[0].get('fuse_walls', []):
            json_coords = json_wall[0]
            json_size = json_wall[1]

            # Ignore each part of border wall (we make it automatically).
            if (
                json_coords[0] == JSON_BORDER_WALL_MIN_X or
                json_coords[0] == JSON_BORDER_WALL_MAX_X or
                json_coords[1] == JSON_BORDER_WALL_MIN_Z or
                json_coords[1] == JSON_BORDER_WALL_MAX_Z
            ):
                continue

            # Create the MCS wall object.
            wall_object = _create_object(
                'fuse_wall_',
                'cube',
                FUSE_WALL_OBJECT_MATERIAL,
                FUSE_WALL_OBJECT_HEIGHT,
                FUSE_WALL_OBJECT_SIZE,
                json_coords,
                json_size,
                unit_size
            )
            wall_object['kinematic'] = True
            wall_object['structure'] = True

            # Adjust the show step to sync with the trial step.
            wall_object['shows'][0]['stepBegin'] = step

            # Don't add duplicate walls.
            coords_property = str(json_coords[0]) + '_' + str(json_coords[1])
            if coords_property not in original_coords_to_object:
                fuse_wall_object_list.append(wall_object)
                # Save the reference to this wall object for later use.
                original_coords_to_object[coords_property] = wall_object

        for frame_index, frame in enumerate(trial):
            existing_coords = {}
            # Identify the fuse walls that still exist in this frame.
            for json_wall in frame.get('fuse_walls', []):
                json_coords = json_wall[0]
                existing_coords[
                    str(json_coords[0]) + '_' + str(json_coords[1])
                ] = True
            # Remove the fuse walls that don't exist in this frame.
            for coords, wall_object in original_coords_to_object.items():
                if (
                    coords not in existing_coords and
                    'hides' not in wall_object
                ):
                    wall_object['hides'] = [{
                        'stepBegin': step + frame_index
                    }]

    return fuse_wall_object_list


def _create_goal_object_list(
    trial_list: List[List[Dict[str, Any]]],
    goal_object_config_list: List[ObjectConfigWithMaterial],
    agent_start_bounds: ObjectBounds,
    filename_prefix: str,
    unit_size: Tuple[float, float]
) -> List[Dict[str, Any]]:
    """Create and return the MCS scene's goal object list using the given
    trial list from the JSON file data."""

    goal_object_list = []

    # Map the JSON icon of each goal object to an index in the MCS object list
    # because sometimes they change positions in the JSON 'objects' list.
    icon_to_index = {}

    # Retrieve the objects data from the first frame of the first trial.
    # Assume the number of objects will never change, and the objects will
    # never change shape/color.
    for index, json_object in enumerate(trial_list[0][0]['objects']):
        json_coords = json_object[0]
        json_radius = json_object[1]
        json_icon = json_object[2]
        json_size = [json_radius * 2, json_radius * 2]
        icon_to_index[json_icon] = index

        # Create the MCS goal object.
        config_with_material = goal_object_config_list[index]
        dimensions = OBJECT_DIMENSIONS[config_with_material.object_type]
        # Multiply the object's scale based on its JSON radius and unit size.
        factor = json_radius * 2 * min(unit_size[0], unit_size[1])
        scale_xz = config_with_material.scale_xz * factor
        scale_y = config_with_material.scale_y * factor
        center_y = dimensions.center_y * config_with_material.scale_y * factor
        goal_object = _create_object(
            'object_',
            config_with_material.object_type,
            config_with_material.material,
            [center_y, scale_y],
            [scale_xz, scale_xz],
            json_coords,
            json_size,
            unit_size
        )
        # Set kinematic to avoid awkward shifting due to collision issues.
        goal_object['kinematic'] = True
        # Set physics so this goal's info is returned in the oracle metadata.
        goal_object['physics'] = True
        goal_object['debug'][
            tags.SCENE.UNTRAINED_SHAPE
        ] = config_with_material.untrained
        goal_object_list.append(goal_object)

        # Add the object's bounds for each other frame of the first trial.
        for _ in range(0, len(trial_list[0])):
            goal_object['debug']['boundsAtStep'].append(
                goal_object['shows'][-1]['boundingBox']
            )

    # Find the step for the start of the second trial.
    # Assume scenes will have more than one trial.
    step = _identify_trial_index_starting_step(1, trial_list)

    # Add data for each object's new position to each trial's start step.
    # Assume objects only change in position across trials (not frames).
    for trial in trial_list[1:]:
        for json_object in trial[0]['objects']:
            json_coords = json_object[0]
            json_radius = json_object[1]
            json_icon = json_object[2]
            json_size = [json_radius * 2, json_radius * 2]

            # Find the MCS object corresponding to this JSON object's icon.
            goal_object = goal_object_list[icon_to_index[json_icon]]

            # Move the object to its new position for the trial.
            goal_object['shows'].append(_create_show(
                step,
                goal_object['type'],
                goal_object['debug']['configHeight'],
                goal_object['debug']['configSize'],
                json_coords,
                json_size,
                unit_size
            ))
            # Add the object's bounds for each frame of the trial.
            for _ in range(0, len(trial) + 1):
                goal_object['debug']['boundsAtStep'].append(
                    goal_object['shows'][-1]['boundingBox']
                )
        # Add 1 for the EndHabituation action step at the end of the trial.
        step += len(trial) + 1

    for goal_object in goal_object_list:
        for index, show in enumerate(goal_object['shows']):
            # We can't have the object's position on top of the agent's start
            # position or the agent and object will collide. This can happen
            # if the 2D icons overlap themselves in the original data.
            if sat_entry(
                agent_start_bounds.box_xz,
                show['boundingBox'].box_xz
            ):
                raise SceneException(
                    f'Cannot convert {filename_prefix} because an object is '
                    f'on the agent\'s home in trial {index + 1}')
            # Remove the scale from each element in 'shows' except for the
            # first, or it will really mess up the simulation.
            if index > 0:
                del show['scale']

    return goal_object_list


def _create_home_object(
    trial_list: List[List[Dict[str, Any]]],
    unit_size: Tuple[float, float]
) -> Dict[str, Any]:
    """Create and return the MCS scene's home object using the given trial
    list from the JSON file data."""

    # Retrieve the home data from the first frame of the first trial.
    # Assume only one home and the home will never change.
    json_home = trial_list[0][0]['home']
    json_coords = json_home[0]
    json_radius = json_home[1]
    json_size = [json_radius * 2, json_radius * 2]

    # Create the MCS home object.
    home_object = _create_object(
        'home_',
        'cube',
        HOME_OBJECT_MATERIAL,
        HOME_OBJECT_HEIGHT,
        HOME_OBJECT_SIZE,
        json_coords,
        json_size,
        unit_size
    )
    home_object['kinematic'] = True
    home_object['structure'] = True
    return home_object


def _create_key_object(
    trial_list: List[List[Dict[str, Any]]],
    unit_size: Tuple[float, float],
    agent_height: float
) -> Optional[Dict[str, Any]]:
    """Create and return the MCS scene's key object using the given trial
    list from the JSON file data."""

    # Retrieve the key data from the first frame of the first trial.
    # Assume the number of keys will never change, and the keys will
    # never change shape/color.
    # Old prop: 'key' ... New prop: 'pin'
    json_key_list = trial_list[0][0].get('pin', trial_list[0][0].get('key'))
    if not json_key_list:
        return None
    json_key = json_key_list[0]
    json_coords = json_key[0]
    json_radius = json_key[1]
    json_size = [json_radius * 2, json_radius * 2]

    # Create the MCS key object.
    key_object = _create_object(
        'key_',
        KEY_OBJECT_TYPE,
        KEY_OBJECT_MATERIAL,
        KEY_OBJECT_HEIGHT,
        KEY_OBJECT_SIZE,
        json_coords,
        json_size,
        unit_size
    )

    # Remove the object's first appearance (we will override it later).
    key_object['shows'] = []
    key_object['debug']['boundsAtStep'] = []
    key_object['debug']['agentHeight'] = agent_height

    # Move the key on each step as needed.
    for trial_index, trial in enumerate(trial_list):
        _append_each_show_to_object(
            key_object,
            trial,
            _identify_trial_index_starting_step(trial_index, trial_list),
            # Old prop: 'key' ... New prop: 'pin'
            'pin' if 'pin' in trial[0] else 'key',
            unit_size,
            _fix_key_location
        )

    # Override the key object with its correct scale.
    key_object['shows'][0]['scale'] = {
        'x': KEY_OBJECT_SIZE[0],
        'y': KEY_OBJECT_HEIGHT[1],
        'z': KEY_OBJECT_SIZE[1]
    }

    # In Eval 4, the key object had structure=True, but that was a bug, and has
    # been fixed for Eval 5 and beyond.

    # Set kinematic to avoid awkward shifting due to collision issues.
    key_object['kinematic'] = True
    # Set physics so this key's info is returned in the oracle metadata.
    key_object['physics'] = True

    key_object['hides'] = []

    return key_object


def _create_lock_wall_object_list(
    trial_list: List[List[Dict[str, Any]]],
    key_object: Dict[str, Any],
    unit_size: Tuple[float, float]
) -> Dict[str, Any]:
    """Create and return the MCS scene's green lock wall object list (used in
    the instrumental action scenes) using the given trial list from the JSON
    file data."""

    lock_wall_object_list = []

    prop_is_lock = False

    # Retrieve the complete lock wall data from each frame of each trial.
    # Assume each trial will have a new lock wall, and the lock wall will be
    # hidden on a specific frame.
    for trial_index, trial in enumerate(trial_list):
        # Identify the step on which this trial starts.
        step = _identify_trial_index_starting_step(trial_index, trial_list)

        # Generate the lock wall used in this trial. Assume only one exists.
        # Old prop: 'lock' ... New prop: 'key'
        # Look for 'lock' FIRST, else it may try to use the old 'key' property.
        json_lock_list = trial[0].get('lock', trial[0].get('key'))
        if json_lock_list:
            prop_is_lock = ('lock' in trial[0])
            json_lock = json_lock_list[0]
            json_coords = json_lock[0]
            json_radius = json_lock[1]
            json_icon = json_lock[2]
            json_size = [json_radius * 2, json_radius * 2]

            # Create the MCS lock wall object.
            lock_object = _create_object(
                'lock_',
                'lock_wall',
                LOCK_WALL_OBJECT_MATERIAL,
                LOCK_WALL_OBJECT_HEIGHT,
                LOCK_WALL_OBJECT_SIZE,
                json_coords,
                json_size,
                unit_size
            )
            lock_object['kinematic'] = True
            lock_object['structure'] = True

            # Rotate the lock based on the JSON icon.
            rotation_y = 0
            if json_icon.endswith('slot90.png'):
                pass
            elif json_icon.endswith('slot180.png'):
                rotation_y = 270
            elif json_icon.endswith('slot270.png'):
                rotation_y = 180
            elif json_icon.endswith('slot0.png'):
                rotation_y = 90
            else:
                raise SceneException(
                    f'Lock is unexpected icon: {json_icon}'
                )
            lock_object['shows'][0]['rotation'] = {
                'x': 0,
                'y': rotation_y,
                'z': 0
            }
            lock_object['debug']['boundsAtStep'] = (
                ([None] * step) + [lock_object['shows'][0]['boundingBox']]
            )
            # Adjust the show step to sync with the trial step.
            lock_object['shows'][0]['stepBegin'] = step

            lock_wall_object_list.append(lock_object)

            for frame_index, frame in enumerate(trial):
                # For each frame in this trial, either repeat the lock wall
                # object's original boundsAtStep, or add a None if the lock
                # is hidden at that frame.
                # Old prop: 'lock' ... New prop: 'key'
                if not frame.get('lock' if prop_is_lock else 'key'):
                    lock_object['debug']['boundsAtStep'].append(None)
                    if 'hides' not in lock_object:
                        # Hide the lock wall if it doesn't exist in this frame.
                        lock_object['hides'] = [{
                            'stepBegin': step + frame_index
                        }]
                        key_object['hides'].append({
                            'stepBegin': step + frame_index
                        })
                else:
                    lock_object['debug']['boundsAtStep'].append(
                        lock_object['debug']['boundsAtStep'][-1]
                    )

        # Add a None to each previous lock's boundsAtStep for each frame in
        # this trial. Always do this even if no lock wall exists in this trial!
        for previous_lock_object in lock_wall_object_list:
            total = step + len(trial) + 1
            previous_lock_object['debug']['boundsAtStep'] = (
                previous_lock_object['debug']['boundsAtStep'] + ([None] * (
                    total - len(previous_lock_object['debug']['boundsAtStep'])
                ))
            )

    return lock_wall_object_list


def _create_object(
    id_prefix: str,
    object_type: str,
    object_material: Tuple[str, str],
    object_height: Tuple[float, float],
    object_size: Tuple[float, float],
    json_coords: Tuple[int, int],
    json_size: Tuple[int, int],
    unit_size: Tuple[float, float],
    rotation_y: int = 0
) -> Dict[str, Any]:
    """Create and return an MCS object using the given data."""
    mcs_object = {
        'id': id_prefix + str(uuid.uuid4()),
        'type': object_type,
        'materials': [object_material.material],
        'debug': {
            'color': object_material.color,
            'info': object_material[1] + [object_type],
            # Save the object's height and size data for future use.
            'configHeight': object_height,
            'configSize': object_size,
            'configRotation': rotation_y
        },
        'shows': [_create_show(
            0,
            object_type,
            object_height,
            object_size,
            json_coords,
            json_size,
            unit_size,
            rotation_y
        )]
    }
    dimensions = OBJECT_DIMENSIONS[object_type]
    scale = mcs_object['shows'][0]['scale']
    mcs_object['debug']['dimensions'] = {
        'x': dimensions.x * scale['x'],
        'y': dimensions.y * scale['y'],
        'z': dimensions.z * scale['z']
    }
    mcs_object['debug']['info'].append(' '.join(mcs_object['debug']['info']))
    mcs_object['debug']['boundsAtStep'] = [
        mcs_object['shows'][0]['boundingBox']
    ]
    return mcs_object


def _create_scene(
    starter_scene: Scene,
    goal_template: Dict[str, Any],
    agent_object_config_list: List[ObjectConfigWithMaterial],
    goal_object_config_list: List[ObjectConfigWithMaterial],
    trial_list: List[List[Dict[str, Any]]],
    filename_prefix: str,
    platform_material: MaterialTuple,
    is_expected: bool
) -> Scene:
    """Create and return the MCS scene using the given templates, trial
    list, and expectedness answer from the JSON file data."""

    scene = copy.deepcopy(starter_scene)
    scene.version = 3
    scene.isometric = True

    scene.goal = copy.deepcopy(goal_template)
    scene.goal['action_list'] = _create_action_list(trial_list)
    scene.goal['habituation_total'] = len(trial_list) - 1
    scene.goal['last_step'] = len(scene.goal['action_list'])
    scene.goal['answer'] = {
        'choice': EXPECTED if is_expected else UNEXPECTED
    }

    unit_size = _retrieve_unit_size(trial_list)
    wall_object_list = _create_wall_object_list(trial_list, unit_size)
    agent_object_list = _create_agent_object_list(
        trial_list,
        agent_object_config_list,
        unit_size
    )
    # Identify the primary agent.
    agent_object = agent_object_list[0]
    # Assume the primary agent is the only one moving around.
    agent_start_bounds = agent_object['shows'][0]['boundingBox']
    goal_object_list = _create_goal_object_list(
        trial_list,
        goal_object_config_list,
        agent_start_bounds,
        filename_prefix,
        unit_size
    )
    home_object = _create_home_object(trial_list, unit_size)
    # Assume the primary agent is the only one that can hold the key.
    agent_height = agent_start_bounds.max_y
    key_object = _create_key_object(trial_list, unit_size, agent_height)
    lock_wall_list = _create_lock_wall_object_list(
        trial_list,
        key_object,
        unit_size
    )
    target, non_target_list = _identify_target_object(
        trial_list,
        goal_object_list
    )

    _remove_intersecting_agent_steps(
        agent_object_list,
        goal_object_list + lock_wall_list
    )
    _remove_extraneous_object_show(
        agent_object_list + [key_object] if key_object else [],
        trial_list
    )
    _move_agent_past_lock_location(agent_object_list, lock_wall_list)
    _move_agent_adjacent_to_goal(
        agent_object_list,
        goal_object_list,
        trial_list
    )

    # If the agent is carrying the key on this step, move the key to be
    # centered directly above the agent.
    for key_show in (key_object['shows'] if key_object else []):
        if key_show['position']['y'] >= agent_height:
            position_x = KEY_OBJECT_ROTATION_Y[
                key_show['rotationProperty']
            ]['position_x']
            position_z = KEY_OBJECT_ROTATION_Y[
                key_show['rotationProperty']
            ]['position_z']
            agent_show = None
            for next_agent_show in agent_object['shows']:
                if next_agent_show['stepBegin'] > key_show['stepBegin']:
                    break
                agent_show = next_agent_show
            key_show['position']['x'] = (
                agent_show['position']['x'] + (position_x / 2.0)
            )
            key_show['position']['z'] = (
                agent_show['position']['z'] + (position_z / 2.0)
            )
        del key_show['rotationProperty']

    # Round object float properties to reduce the size of output scene files.
    for mcs_object in (
        agent_object_list + ([key_object] if key_object else []) + [target] +
        non_target_list
    ):
        for mcs_show in mcs_object['shows']:
            for a in ['x', 'y', 'z']:
                mcs_show['position'][a] = round(mcs_show['position'][a], 4)

    platform = structures.create_platform(
        position_x=4,
        position_z=-4,
        rotation_y=0,
        scale_x=0.75,
        scale_y=3,
        scale_z=0.75,
        room_dimension_y=10,
        material_tuple=platform_material
    )

    # Set distinct random materials for the floor and room walls.
    # Ensure they don't match the color of any important objects.
    excluded_colors = [
        color for mcs_object in (agent_object_list + goal_object_list)
        for color in mcs_object['debug']['color']
    ]
    scene.ceiling_material = CEILING_MATERIAL.material
    floor_choices = [material_tuple for material_tuple in FLOOR_MATERIALS if (
        material_tuple.color[0] not in excluded_colors
    )]
    floor_choice = random.choice(floor_choices)
    scene.floor_material = floor_choice.material
    scene.debug['floorColors'] = floor_choice.color
    wall_choices = [material_tuple for material_tuple in WALL_MATERIALS if (
        material_tuple.color[0] not in excluded_colors and
        material_tuple.material != floor_choice.material
    )]
    wall_choice = random.choice(wall_choices)
    scene.wall_material = wall_choice.material
    scene.debug['wallColors'] = wall_choice.color

    role_to_object_list = {}
    role_to_object_list[tags.ROLES.AGENT] = agent_object_list
    role_to_object_list[tags.ROLES.HOME] = [home_object]
    role_to_object_list[tags.ROLES.KEY] = [key_object] if key_object else []
    role_to_object_list[tags.ROLES.NON_TARGET] = non_target_list
    role_to_object_list[tags.ROLES.STRUCTURAL] = [platform]
    role_to_object_list[tags.ROLES.TARGET] = [target]
    role_to_object_list[tags.ROLES.WALL] = wall_object_list + lock_wall_list

    scene = update_scene_objects(scene, role_to_object_list)
    return scene


def _create_show(
    begin_frame: int,
    object_type: str,
    object_height: Tuple[float, float],
    object_size: Tuple[float, float],
    json_coords: Tuple[int, int],
    json_size: Tuple[int, int],
    unit_size: Tuple[float, float],
    rotation_y: int = 0
) -> Dict[str, Any]:
    """Create and return an MCS object's 'shows' element using the given
    data."""
    mcs_show = {
        'stepBegin': begin_frame,
        'position': {
            'x': GRID_MIN_X + (
                (json_coords[0] + (json_size[0] / 2)) * unit_size[0]
            ),
            'y': object_height[0],
            'z': GRID_MIN_Z + (
                (json_coords[1] + (json_size[1] / 2)) * unit_size[1]
            )
        },
        'rotation': {'x': 0, 'y': rotation_y, 'z': 0},
        'scale': {
            'x': object_size[0],
            'y': object_height[1],
            'z': object_size[1]
        }
    }
    dimensions = OBJECT_DIMENSIONS[object_type]
    mcs_show['boundingBox'] = geometry.create_bounds(
        dimensions={
            'x': mcs_show['scale']['x'] * dimensions.x,
            'y': mcs_show['scale']['y'] * dimensions.y,
            'z': mcs_show['scale']['z'] * dimensions.z
        },
        offset={'x': 0, 'y': 0, 'z': 0},
        position=mcs_show['position'],
        rotation=mcs_show['rotation'],
        standing_y=(mcs_show['scale']['y'] * dimensions.y / 2.0)
    )
    return mcs_show


def _create_static_wall_object_list(
    trial_list: List[List[Dict[str, Any]]],
    unit_size: Tuple[float, float]
) -> List[Dict[str, Any]]:
    """Create and return the MCS scene's black static wall object list using
    the given trial list from the JSON file data."""

    static_wall_object_list = []

    step = 0
    for trial_index, trial in enumerate(trial_list):
        # Retrieve the wall data list from the first frame of each trial.
        json_wall_list = trial[0]['walls']

        # Only create new walls for the first trial or if they've changed.
        if (
            trial_index == 0 or
            json_wall_list != trial_list[trial_index - 1][0]['walls']
        ):
            # Assume the walls have changed, so hide any previous walls.
            for wall_object in static_wall_object_list:
                if not wall_object.get('hides'):
                    wall_object['hides'] = [{
                        'stepBegin': step
                    }]

            for json_wall in json_wall_list:
                json_coords = json_wall[0]
                json_size = json_wall[1]

                # Ignore each part of border wall (we make it automatically).
                if (
                    json_coords[0] == JSON_BORDER_WALL_MIN_X or
                    json_coords[0] == JSON_BORDER_WALL_MAX_X or
                    json_coords[1] == JSON_BORDER_WALL_MIN_Z or
                    json_coords[1] == JSON_BORDER_WALL_MAX_Z
                ):
                    continue

                # Create the MCS wall object and add it to the list.
                wall_object = _create_object(
                    'wall_',
                    'cube',
                    WALL_OBJECT_MATERIAL,
                    WALL_OBJECT_HEIGHT,
                    WALL_OBJECT_SIZE,
                    json_coords,
                    json_size,
                    unit_size
                )
                wall_object['kinematic'] = True
                wall_object['structure'] = True
                # Adjust the show step to sync with the trial step.
                wall_object['shows'][0]['stepBegin'] = step
                static_wall_object_list.append(wall_object)

        # Add 1 for the EndHabituation action step at the end of the trial.
        step += len(trial) + 1

    return static_wall_object_list


def _create_trial_frame_list(
    trial: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """Return all the frames in the given trial that we want to keep in the
    final MCS scene using the agent's movement. Skip about half of the frames
    to make the MCS simulation a bit quicker."""

    frame_list = []
    starting_coords = {}
    previous_coords = {}
    json_property_list = ['agent', 'fuse_walls', 'key', 'lock', 'pin']
    for json_property in json_property_list:
        starting_coords[json_property] = trial[0].get(json_property)
        previous_coords[json_property] = trial[0].get(json_property)
    starting_frame_count = STARTING_STEP_WAIT_TIME
    paused_frame_count = PAUSED_STEP_WAIT_TIME
    defuse_frame_count = 5
    skip_next = False

    for index, frame in enumerate(trial):
        # Keep or remove frames based on the movement of the agents/objects.
        coords = {}
        for json_property in json_property_list:
            coords[json_property] = frame.get(json_property)
        # Only keep a specific number of the trial's starting frames.
        if coords['agent'] == starting_coords['agent']:
            if starting_frame_count > 0:
                frame_list.append(frame)
                starting_frame_count -= 1
            continue
        # Only keep a specific number of the trial's agent-is-paused frames.
        if coords['agent'] == previous_coords['agent']:
            is_repeated = True
            for json_property in json_property_list:
                if coords[json_property] != previous_coords[json_property]:
                    is_repeated = False
                    break
            # If a separate object is changing, don't skip this frame.
            if is_repeated:
                if paused_frame_count > 0:
                    frame_list.append(frame)
                    paused_frame_count -= 1
                continue
            else:
                skip_next = False
        # Reset the paused frame count if an object moves again.
        paused_frame_count = PAUSED_STEP_WAIT_TIME
        if coords['fuse_walls'] != previous_coords['fuse_walls']:
            only_fuse_wall = len(coords['fuse_walls'])
            for json_property in json_property_list:
                if json_property == 'fuse_walls':
                    continue
                if coords[json_property] != previous_coords[json_property]:
                    only_fuse_wall = False
                    break
            if only_fuse_wall:
                if defuse_frame_count == DEFUSE_STEP_SKIP_TIME:
                    frame_list.append(frame)
                    defuse_frame_count = 0
                defuse_frame_count += 1
                continue
            # If all the fuse walls are gone now, add a few steps of buffer
            # before the agent starts moving by copying the frame.
            if (
                len(coords['fuse_walls']) == 0 and
                len(previous_coords['fuse_walls']) > 0
            ):
                for _ in range(POST_DEFUSE_WAIT_TIME):
                    frame_list.append(frame)
        defuse_frame_count = DEFUSE_STEP_SKIP_TIME
        # Skip this frame if we used the previous frame.
        # Keep it if it's the last frame of the trial.
        if skip_next and index < (len(trial) - 1):
            skip_next = False
            continue
        # Else keep this frame.
        frame_list.append(frame)
        for json_property in json_property_list:
            previous_coords[json_property] = coords[json_property]
        skip_next = True

    return frame_list


def _create_wall_object_list(
    trial_list: List[List[Dict[str, Any]]],
    unit_size: Tuple[float, float]
) -> List[Dict[str, Any]]:
    """Create and return the MCS scene's wall object list using the given
    trial list from the JSON file data."""
    fuse_wall_list = _create_fuse_wall_object_list(trial_list, unit_size)
    static_wall_list = _create_static_wall_object_list(trial_list, unit_size)
    for name, position, size in [
        ('wall_front', (0, 2.25), (5, WALL_OBJECT_SIZE[1])),
        ('wall_back', (0, -2.25), (5, WALL_OBJECT_SIZE[1])),
        ('wall_left', (-2.25, 0), (WALL_OBJECT_SIZE[0], 4)),
        ('wall_right', (2.25, 0), (WALL_OBJECT_SIZE[0], 4))
    ]:
        wall_object = {
            'id': name,
            'type': 'cube',
            'materials': [WALL_OBJECT_MATERIAL[0]],
            'shows': [{
                'stepBegin': 0,
                'position': {
                    'x': position[0],
                    'y': WALL_OBJECT_HEIGHT[0],
                    'z': position[1]
                },
                'rotation': {
                    'x': 0,
                    'y': 0,
                    'z': 0
                },
                'scale': {
                    'x': size[0],
                    'y': WALL_OBJECT_HEIGHT[1],
                    'z': size[1]
                }
            }],
            'kinematic': True,
            'structure': True,
            'debug': {
                'info': WALL_OBJECT_MATERIAL[1] + ['cube'],
            }
        }
        wall_object['debug']['info'].append(
            ' '.join(wall_object['debug']['info'])
        )
        wall_object['shows'][0]['boundingBox'] = geometry.create_bounds(
            dimensions=wall_object['shows'][0]['scale'],
            offset={'x': 0, 'y': 0, 'z': 0},
            position=wall_object['shows'][0]['position'],
            rotation=wall_object['shows'][0]['rotation'],
            standing_y=(wall_object['shows'][0]['scale']['y'] / 2.0)
        )
        static_wall_list.append(wall_object)
    return static_wall_list + fuse_wall_list


def _fix_key_location(
    trial_start_step: int,
    json_key: Dict[str, Any],
    key_object: Dict[str, Any]
) -> Dict[str, Any]:
    """Update the given key object's location on a specific step (frame) in the
    current trial using the given JSON key data. Used as the on_step_callback
    parameter to _append_each_show_to_object on key objects."""

    # Rotate the key based on the JSON icon.
    json_icon = json_key[2]
    rotation_property = None
    if json_icon.endswith('triangle90.png'):
        rotation_property = 'negative_z'
    elif json_icon.endswith('triangle180.png'):
        rotation_property = 'positive_x'
    elif json_icon.endswith('triangle270.png'):
        rotation_property = 'positive_z'
    elif json_icon.endswith('triangle0.png'):
        rotation_property = 'negative_x'
    else:
        raise SceneException(f'Key is unexpected icon: {json_icon}')
    this_show = key_object['shows'][-1]
    this_show['rotation'] = {
        'x': KEY_OBJECT_ROTATION_X,
        'y': KEY_OBJECT_ROTATION_Y[rotation_property]['rotation_y'],
        'z': KEY_OBJECT_ROTATION_Z
    }
    # Save the rotation property for use in _create_key_object
    this_show['rotationProperty'] = rotation_property

    # Adjust the key's location based on its rotation (since the triangle isn't
    # centered on the X/Z axis). Also adjust its bounding box accordingly.
    this_show['position']['x'] += (
        KEY_OBJECT_ROTATION_Y[rotation_property]['position_x']
    )
    this_show['position']['z'] += (
        KEY_OBJECT_ROTATION_Y[rotation_property]['position_z']
    )
    dimensions_x = KEY_OBJECT_ROTATION_Y[rotation_property]['dimensions_x']
    dimensions_z = KEY_OBJECT_ROTATION_Y[rotation_property]['dimensions_z']
    this_show['boundingBox'] = geometry.create_bounds(
        dimensions={
            'x': dimensions_x,
            'y': KEY_OBJECT_HEIGHT[1],
            'z': dimensions_z
        },
        offset={'x': 0, 'y': 0, 'z': 0},
        position={
            'x': this_show['position']['x'] - (dimensions_x / 2.0),
            'y': this_show['position']['y'],
            'z': this_show['position']['z'] - (dimensions_z / 2.0)
        },
        rotation=this_show['rotation'],
        standing_y=(KEY_OBJECT_HEIGHT[1] / 2.0)
    )

    # Adjust the key's height to above the agent when it starts moving.
    if len(key_object['shows']) > 1:
        previous_show = key_object['shows'][-2]
        if (
            previous_show['position']['x'] != this_show['position']['x'] or
            previous_show['position']['z'] != this_show['position']['z']
        ) and not (
            previous_show['stepBegin'] < trial_start_step and
            this_show['stepBegin'] >= trial_start_step
        ):
            this_show['position']['y'] += key_object['debug']['agentHeight']
    return key_object


def _identify_target_object(
    trial_list: List[List[Dict[str, Any]]],
    object_list: List[Dict[str, Any]]
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    # TODO
    # return target, non_target_list
    return object_list[0], object_list[1:]


def _identify_trial_index_starting_step(
    index: int,
    trial_list: List[List[Dict[str, Any]]]
) -> int:
    """Return the MCS step at the start of the trial with the given
    index."""
    step = 0
    for prior_index in range(0, index):
        # Add 1 for the EndHabituation action step at the end of the trial.
        step += len(trial_list[prior_index]) + 1
    return step


def _move_agent_past_lock_location(
    agent_object_list: List[Dict[str, Any]],
    lock_wall_list: Dict[str, Any]
) -> None:
    """Adjust the agent's movement onto and away from the lock space before
    and after inserting the key and removing the fuse walls."""
    # Assume only one agent in instrumental action scenes.
    agent_object = agent_object_list[0]
    # Assume only one lock per trial, and each lock is only hidden one time.
    for lock_object in lock_wall_list:
        # Sometimes the agent can just ignore the lock, so it never disappears.
        if 'hides' not in lock_object:
            continue
        # Find the first agent movement after the lock is removed.
        for index, show in enumerate(agent_object['shows']):
            if show['stepBegin'] > lock_object['hides'][0]['stepBegin']:
                break
        # This movement will depict the agent teleporting to the center of the
        # lock's now-removed position, which we don't want, so just delete it.
        del agent_object['shows'][index]
        # Remove each following movement that would also be within the lock's
        # position. We'll replace the removed movement next.
        remove_list = []
        lock_bounds = lock_object['debug']['boundsAtStep'][
            lock_object['hides'][0]['stepBegin'] - 1
        ]
        for index_2, target_show in enumerate(agent_object['shows'][(index):]):
            if sat_entry(
                target_show['boundingBox'].box_xz,
                lock_bounds.box_xz
            ):
                remove_list.append(index + index_2)
            else:
                # Break once the movement exits the lock's position.
                break
        remove_list.reverse()
        for i in remove_list:
            del agent_object['shows'][i]
        # Identify the agent's positions before and after moving through the
        # lock's position, so we can replace the movement between them.
        position_x = agent_object['shows'][index - 1]['position']['x']
        position_z = agent_object['shows'][index - 1]['position']['z']
        target_position_x = target_show['position']['x']
        target_position_z = target_show['position']['z']
        # Identify the number of steps that the movement should take (longer
        # distances will take more steps).
        distance = math.sqrt(
            (target_position_x - position_x)**2 +
            (target_position_z - position_z)**2
        )
        # We expect the distance to be at most 0.7, which is the diagonal of a
        # 0.5 x 0.5 grid cell.
        move_time = min(
            max(1, int(round(distance / math.sqrt(2) * 10))),
            POST_DEFUSE_WAIT_TIME
        )
        # Divide the movement over the number of steps.
        fragment_x = (target_position_x - position_x) / move_time
        fragment_z = (target_position_z - position_z) / move_time
        # Adjust the existing show for the 1st fragment of movement as needed.
        target_show['stepBegin'] -= (move_time - 1)
        target_show['position']['x'] = position_x + fragment_x
        target_show['position']['z'] = position_z + fragment_z
        # Insert a new show for each successive fragment of movement.
        for amount in range(1, move_time):
            new_show = copy.deepcopy(target_show)
            new_show['stepBegin'] += amount
            new_show['position']['x'] += (fragment_x * amount)
            new_show['position']['z'] += (fragment_z * amount)
            agent_object['shows'].insert(index + amount, new_show)


def _move_agent_adjacent_to_goal(
    agent_object_list: List[Dict[str, Any]],
    goal_object_list: List[Dict[str, Any]],
    trial_list: List[List[Dict[str, Any]]]
) -> None:
    """Ensure the agent is directly adjacent to its goal in each trial."""
    # Record the starting step of each trial for future use.
    trial_index_to_step = {}
    for trial_index in range(len(trial_list) + 1):
        step = _identify_trial_index_starting_step(trial_index, trial_list)
        trial_index_to_step[trial_index] = step

    for trial_index in range(len(trial_list)):
        first_step = trial_index_to_step[trial_index]
        final_step = trial_index_to_step[trial_index + 1] - 1

        for agent_object in agent_object_list:
            # Identify the agent's final "show" of this trial.
            agent_show = None
            for show in agent_object['shows']:
                if first_step <= show['stepBegin'] <= final_step:
                    agent_show = show
                if final_step < show['stepBegin']:
                    break

            # If the agent is hidden in this trial, skip it.
            if not agent_show:
                continue

            agent_poly = agent_show['boundingBox'].polygon_xz

            # Calculate the distance from the agent to each goal object.
            goals_with_distances = []
            for goal_object in goal_object_list:
                # Identify the goal object's "show" in this trial.
                goal_show = None
                for show in goal_object['shows']:
                    if first_step <= show['stepBegin'] <= final_step:
                        goal_show = show
                    if final_step < show['stepBegin']:
                        break

                # If the goal object does not show in this trial, skip it.
                if not goal_show:
                    continue

                # If the agent's too far away from this goal object, skip it.
                poly = goal_show['boundingBox'].polygon_xz
                distance = agent_poly.distance(poly)
                if distance > 0.5:
                    continue
                goals_with_distances.append((distance, goal_show))

            # If the agent's not near any goal in this trial, skip it.
            if not goals_with_distances:
                continue

            goal_show = sorted(goals_with_distances, key=lambda x: x[0])[0][1]
            goal_poly = goal_show['boundingBox'].polygon_xz

            # Find the nearest distance from the agent to the goal.
            agent_point, goal_point = ops.nearest_points(agent_poly, goal_poly)
            diff_x = goal_point.coords[0][0] - agent_point.coords[0][0]
            diff_z = goal_point.coords[0][1] - agent_point.coords[0][1]

            # Move the agent directly adjacent to the goal.
            agent_show['position']['x'] += diff_x
            agent_show['position']['z'] += diff_z
            agent_dimensions = agent_object['debug']['dimensions']
            agent_show['boundingBox'] = geometry.create_bounds(
                dimensions={
                    'x': agent_dimensions['x'],
                    'y': agent_dimensions['y'],
                    'z': agent_dimensions['z']
                },
                offset={'x': 0, 'y': 0, 'z': 0},
                position=agent_show['position'],
                rotation=agent_show['rotation'],
                standing_y=(agent_dimensions['y'] / 2.0)
            )
            bounds_at_step = agent_object['debug']['boundsAtStep']
            for bounds_index in range(agent_show['stepBegin'], final_step):
                if bounds_index >= len(bounds_at_step):
                    break
                bounds_at_step[bounds_index] = agent_show['boundingBox']


def _remove_extraneous_object_show(
    object_list: List[Dict[str, Any]],
    trial_list: List[List[Dict[str, Any]]]
) -> None:
    """Remove each moving object's 'shows' array element that is the same as
    its previous array element, since they aren't needed, and we can therefore
    reduce the size of the JSON output file. Assume that each object should
    always be shown at the start of each trial."""
    starting_step_list = [
        _identify_trial_index_starting_step(trial_index, trial_list)
        for trial_index in range(len(trial_list))
    ]
    for mcs_object in object_list:
        mcs_object['debug']['extraneousSteps'] = []
        show_list = mcs_object['shows'][:1]
        for show in mcs_object['shows'][1:]:
            # Ignore it if the position/rotation are the same as the previous.
            if (
                show['position'] != show_list[-1]['position'] or
                show.get('rotation', {}) != show_list[-1].get('rotation', {})
            ):
                show_list.append(show)
            # Ensure that we show the object at the start of each trial.
            else:
                for starting_step in starting_step_list:
                    if starting_step == show['stepBegin']:
                        show_list.append(show)
                        break
                if show != show_list[-1]:
                    mcs_object['debug']['extraneousSteps'].append(
                        show['stepBegin']
                    )
        mcs_object['shows'] = show_list


def _remove_intersecting_agent_steps(
    agent_object_list: List[Dict[str, Any]],
    other_object_list: List[Dict[str, Any]]
) -> None:
    """Remove each agent object's step that intersects with any goal object's
    location at that step, since sometimes the agent moves a little too close
    to the goal object."""
    for agent_object in agent_object_list:
        remove_step_list = []
        for step, agent_bounds in enumerate(
            agent_object['debug']['boundsAtStep']
        ):
            if not agent_bounds:
                # If the agent is hidden on this step, skip it.
                continue
            for other_object in other_object_list:
                object_bounds = other_object['debug']['boundsAtStep'][step]
                if object_bounds and sat_entry(
                    agent_bounds.box_xz,
                    object_bounds.box_xz
                ):
                    remove_step_list.append(step)
        agent_object['shows'] = [
            show for show in agent_object['shows']
            if show['stepBegin'] not in remove_step_list
        ]
        agent_object['debug']['intersectingSteps'] = remove_step_list


def _retrieve_unit_size(
    trial_list: List[List[Dict[str, Any]]]
) -> Tuple[float, float]:
    """Return the unit size of this scene's grid."""
    # Assume the JSON grid size will never change.
    json_grid = trial_list[0][0]['size']
    grid_size_x = (GRID_MAX_X - GRID_MIN_X) / json_grid[0]
    grid_size_z = (GRID_MAX_Z - GRID_MIN_Z) / json_grid[1]
    return [grid_size_x, grid_size_z]


def _save_trials(trial_list: List[List[Dict[str, Any]]], filename_prefix: str):
    """Save the modified JSON list of trials to text file for debugging."""
    with open(f'{filename_prefix}{TRIALS_SUFFIX}', 'w') as output_file:
        for trial_index, trial in enumerate(trial_list):
            step = _identify_trial_index_starting_step(trial_index, trial_list)
            output_file.write(f'TRIAL {trial_index + 1} STEP {step}\n\n')
            for frame_index, frame in enumerate(trial):
                output_file.write(f'FRAME {frame_index + 1}\n')
                output_file.write(f'AGENT {frame["agent"]}\n')
                output_file.write(f'FUSE_WALLS {frame.get("fuse_walls")}\n')
                output_file.write(f'KEY {frame.get("key")}\n')
                output_file.write(f'LOCK {frame.get("lock")}\n')
                output_file.write(f'PIN {frame.get("pin")}\n')
                output_file.write(f'OBJECTS {frame["objects"]}\n')
                output_file.write('\n')


def convert_scene_pair(
    starter_scene: Dict[str, Any],
    goal_template: Dict[str, Any],
    trial_list_expected: List[List[Dict[str, Any]]],
    trial_list_unexpected: List[List[Dict[str, Any]]],
    filename_prefix: str,
    role_to_type: Dict[str, str],
    untrained: bool
) -> List[Dict[str, Any]]:
    """Create and return the pair of MCS scenes using the given templates
    and trial lists from the JSON file data."""

    # Ignore untrained for now.
    untrained = False

    # Create the converted trial lists for both of the scenes. This will
    # remove extraneous frames from all of the trials.
    converted_trial_list_expected = [
        _create_trial_frame_list(trial) for trial in trial_list_expected
    ]

    if SAVE_TRIALS_TO_FILE:
        _save_trials(
            converted_trial_list_expected,
            f'{filename_prefix[(filename_prefix.rfind("/") + 1):]}e'
        )

    # Choose the shape and color of each object in a scene so we can use them
    # in both scenes across the pair so they will always have the same config.
    agent_object_config_list = _choose_config_list(
        converted_trial_list_expected,
        [
            config for config in AGENT_OBJECT_CONFIG_LIST
            if config.untrained == untrained
        ],
        [role_to_type[tags.ROLES.AGENT], role_to_type['second agent']],
        AGENT_OBJECT_MATERIAL_LIST,
        'agent',
        [],
        []
    )
    goal_object_config_list = _choose_config_list(
        converted_trial_list_expected,
        [
            config for config in GOAL_OBJECT_CONFIG_LIST
            if config.untrained == untrained
        ],
        [role_to_type[tags.ROLES.TARGET], role_to_type[tags.ROLES.NON_TARGET]],
        GOAL_OBJECT_MATERIAL_LIST,
        'objects',
        [item.object_type for item in agent_object_config_list],
        [item.material[0] for item in agent_object_config_list]
    )

    # Ensure the two scenes have exactly the same platform material.
    platform_material = random.choice(materials.WALL_MATERIALS)

    logger.info('Generating expected MCS agent scene from JSON data')
    scene_expected = _create_scene(
        starter_scene,
        goal_template,
        agent_object_config_list,
        goal_object_config_list,
        converted_trial_list_expected,
        filename_prefix,
        platform_material,
        is_expected=True
    )
    scenes = [scene_expected]

    # Training datasets will not have any unexpected scenes.
    if trial_list_unexpected:
        logger.info('Generating unexpected MCS agent scene from JSON data')
        converted_trial_list_unexpected = [
            _create_trial_frame_list(trial) for trial in trial_list_unexpected
        ]
        scene_unexpected = _create_scene(
            starter_scene,
            goal_template,
            agent_object_config_list,
            goal_object_config_list,
            converted_trial_list_unexpected,
            filename_prefix,
            platform_material,
            is_expected=False
        )
        # Ensure the two scenes have exactly the same room materials.
        for prop in ['ceilingMaterial', 'floorMaterial', 'wallMaterial']:
            scene_unexpected[prop] = scene_expected[prop]
        for prop in ['floorColors', 'wallColors']:
            scene_unexpected['debug'][prop] = scene_expected['debug'][prop]
        scenes.append(scene_unexpected)

    return scenes
