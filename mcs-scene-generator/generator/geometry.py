import copy
import logging
import math
import random
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
from machine_common_sense.config_manager import Vector3d
from shapely import affinity, geometry, ops

from .definitions import (
    DefinitionDataset,
    ImmutableObjectDefinition,
    ObjectDefinition
)
from .separating_axis_theorem import sat_entry

MAX_TRIES = 50

MAX_REACH_DISTANCE = 1.0
MOVE_DISTANCE = 0.1

FLOOR_FEATURE_BOUNDS_BUFFER = 0.001

PERFORMER_CAMERA_Y = 0.762
PERFORMER_HALF_WIDTH = 0.25
PERFORMER_HEIGHT = 1.25
PERFORMER_MASS = 2
PERFORMER_WIDTH = PERFORMER_HALF_WIDTH * 2.0

MIN_RANDOM_INTERVAL = 0.05
POSITION_DIGITS = 2
VALID_ROTATIONS = (0, 45, 90, 135, 180, 225, 270, 315)

DEFAULT_ROOM_DIMENSIONS = {'x': 10, 'y': 3, 'z': 10}

MAX_OBJECTS_ADJACENT_DISTANCE = 0.5
MIN_OBJECTS_SEPARATION_DISTANCE = 2
MIN_FORWARD_VISIBILITY_DISTANCE = 1.25
MIN_GAP = 0.5

FRONT_WALL_LABEL = "front_wall"
BACK_WALL_LABEL = "back_wall"
LEFT_WALL_LABEL = "left_wall"
RIGHT_WALL_LABEL = "right_wall"

ORIGIN = {
    "x": 0.0,
    "y": 0.0,
    "z": 0.0
}

ORIGIN_LOCATION = {
    'position': {
        'x': 0.0,
        'y': 0.0,
        'z': 0.0
    },
    'rotation': {
        'x': 0.0,
        'y': 0.0,
        'z': 0.0
    }
}


@dataclass
class ObjectBounds():
    """The bounds of a specific object, including its 2D bounding box, 2D
    polygon, and Y range."""
    box_xz: List[Vector3d]
    max_y: float
    min_y: float
    polygon_xz: geometry.Polygon = None

    def __post_init__(self):
        self._update_poly()

    def __eq__(self, other) -> bool:
        return (
            self.box_xz == other.box_xz and self.max_y == other.max_y and
            self.min_y == other.min_y
        )

    def _update_poly(self) -> None:
        for point in self.box_xz:
            for attr in ['x', 'y', 'z']:
                setattr(point, attr, round(getattr(point, attr), 6))
        points = [(point.x, point.z) for point in self.box_xz]
        self.polygon_xz = geometry.Polygon(points)

    def expand_by(self, amount: float) -> None:
        """Expand this bounds by the given amount on both the X and Z axes."""
        self.polygon_xz = self.polygon_xz.buffer(
            amount,
            # Ensure the output polygon is also a rectangle.
            join_style=geometry.JOIN_STYLE.mitre
        )
        self.box_xz = [
            Vector3d(x=point[0], y=0, z=point[1]) for point in
            self.polygon_xz.exterior.coords
        ][:-1]

    def extend_bottom_to_ground(self) -> None:
        """Extend the bottom of this bounds to the ground."""
        # We're not currently saving any height data in the box or polygon,
        # so just update the min_y for now.
        self.min_y = 0

    def is_within_room(self, room_dimensions: Dict[str, float]) -> bool:
        """Return whether this bounds in within the given room dimensions."""
        room_max_x = room_dimensions['x'] / 2.0
        room_max_z = room_dimensions['z'] / 2.0
        return all((
            -room_max_x <= point.x <= room_max_x and
            -room_max_z <= point.z <= room_max_z
        ) for point in self.box_xz)


def __dict_to_vector(data: Dict[str, float]) -> Vector3d:
    return Vector3d(x=data['x'], y=data['y'], z=data['z'])


def create_bounds(
    # TODO MCS-697 MCS-698 Use Vector3d instead of Dict
    dimensions: Dict[str, float],
    offset: Optional[Dict[str, float]],
    position: Dict[str, float],
    rotation: Dict[str, float],
    standing_y: float
) -> ObjectBounds:
    """Creates and returns an ObjectBounds for the object with the given size
    properties in the given location."""
    # TODO MCS-697 MCS-698 Use class props directly instead of converting
    dimensions = __dict_to_vector(dimensions)
    offset = __dict_to_vector(offset) if offset else Vector3d()
    position = __dict_to_vector(position)
    rotation = __dict_to_vector(rotation)

    radian_amount = math.pi * (2 - (rotation.y % 360) / 180.0)

    rotate_sin = math.sin(radian_amount)
    rotate_cos = math.cos(radian_amount)
    x_plus = (dimensions.x / 2.0) + offset.x
    x_minus = -(dimensions.x / 2.0) + offset.x
    z_plus = (dimensions.z / 2.0) + offset.z
    z_minus = -(dimensions.z / 2.0) + offset.z

    a = Vector3d(
        x=position.x + x_plus * rotate_cos - z_plus * rotate_sin,
        y=0,
        z=position.z + x_plus * rotate_sin + z_plus * rotate_cos
    )
    b = Vector3d(
        x=position.x + x_plus * rotate_cos - z_minus * rotate_sin,
        y=0,
        z=position.z + x_plus * rotate_sin + z_minus * rotate_cos
    )
    c = Vector3d(
        x=position.x + x_minus * rotate_cos - z_minus * rotate_sin,
        y=0,
        z=position.z + x_minus * rotate_sin + z_minus * rotate_cos
    )
    d = Vector3d(
        x=position.x + x_minus * rotate_cos - z_plus * rotate_sin,
        y=0,
        z=position.z + x_minus * rotate_sin + z_plus * rotate_cos
    )

    y_min = position.y - standing_y
    y_max = y_min + dimensions.y
    return ObjectBounds(box_xz=[a, b, c, d], max_y=y_max, min_y=y_min)


def random_real(a: float, b: float,
                step: float = MIN_RANDOM_INTERVAL) -> float:
    """Return a random real number N where a <= N <= b and N - a is
    divisible by step."""
    steps = int((b - a) / step)
    try:
        n = random.randint(0, steps)
    except ValueError as e:
        raise ValueError(f'bad args to random_real: ({a}, {b}, {step})', e)
    return a + (n * step)


def random_position_x(room_dimensions: Dict[str, float]) -> float:
    room_max_x = room_dimensions['x'] / 2.0
    return round(random.uniform(-room_max_x, room_max_x), POSITION_DIGITS)


def random_position_z(room_dimensions: Dict[str, float]) -> float:
    room_max_z = room_dimensions['z'] / 2.0
    return round(random.uniform(-room_max_z, room_max_z), POSITION_DIGITS)


def random_rotation() -> float:
    return random.choice(VALID_ROTATIONS)


def calc_obj_pos(
    performer_position: Dict[str, float],
    bounds_list: List[ObjectBounds],
    # TODO MCS-697 Define an ObjectInstance class extending ObjectDefinition.
    definition_or_instance: Union[ObjectDefinition, Dict[str, Any]],
    x_func: Callable[[Dict[str, float]], float] = random_position_x,
    z_func: Callable[[Dict[str, float]], float] = random_position_z,
    rotation_func: Callable[[], float] = None,
    xz_func: Callable[[], Tuple[float, float]] = None,
    room_dimensions: Dict[str, float] = None
) -> Optional[Dict[str, Any]]:
    """Returns new object with rotation & position if we can place the
    object in the frame, None otherwise."""

    # TODO MCS-697 Define an ObjectInstance class extending ObjectDefinition.
    if isinstance(definition_or_instance, dict):
        debug = definition_or_instance['debug']
        dimensions = debug['dimensions']
        offset = debug.get('offset', {'x': 0, 'z': 0})
        position_y = debug.get('positionY', 0)
        rotation = debug.get('rotation', {'x': 0, 'y': 0, 'z': 0})
    else:
        dimensions = vars(definition_or_instance.dimensions)
        offset = vars(definition_or_instance.offset)
        position_y = definition_or_instance.positionY
        rotation = vars(definition_or_instance.rotation)

    tries = 0
    while tries < MAX_TRIES:
        rotation_x = rotation['x']
        rotation_y = rotation['y'] + (
            rotation_func() if rotation_func else random_rotation()
        )
        rotation_z = rotation['z']
        if xz_func is not None:
            new_x, new_z = xz_func(room_dimensions or DEFAULT_ROOM_DIMENSIONS)
        else:
            new_x = x_func(room_dimensions or DEFAULT_ROOM_DIMENSIONS)
            new_z = z_func(room_dimensions or DEFAULT_ROOM_DIMENSIONS)

        if new_x is not None and new_z is not None:
            bounds = create_bounds(
                dimensions=dimensions,
                offset=offset,
                position={'x': new_x, 'y': position_y, 'z': new_z},
                rotation={'x': rotation_x, 'y': rotation_y, 'z': rotation_z},
                standing_y=position_y
            )
            if validate_location_rect(
                bounds,
                performer_position,
                bounds_list,
                room_dimensions or DEFAULT_ROOM_DIMENSIONS
            ):
                break
        tries += 1

    if tries < MAX_TRIES:
        object_location = {
            'rotation': {'x': rotation_x, 'y': rotation_y, 'z': rotation_z},
            'position': {
                'x': new_x,
                'y': position_y,
                'z': new_z
            },
            'boundingBox': bounds
        }
        bounds_list.append(bounds)
        return object_location

    logging.debug(f'could not place object: {definition_or_instance}')
    return None


def position_distance(a: Dict[str, float], b: Dict[str, float]) -> float:
    """Compute the distance between two positions."""
    return math.sqrt((a['x'] - b['x'])**2 + (a['y'] -
                                             b['y'])**2 + (a['z'] - b['z'])**2)


def _get_visible_segment(
    performer_start: Dict[str, Dict[str, float]],
    room_dimensions: Dict[str, float]
) -> geometry.LineString:
    """Get a line segment that should be visible to the performer
    (straight ahead and at least MIN_FORWARD_VISIBILITY_DISTANCE but within
    the room). Return None if no visible segment is possible.
    """
    max_dimension = max(room_dimensions['x'], room_dimensions['z'])
    # make it long enough for the far end to be outside the room
    view_segment = geometry.LineString(
        [[0, MIN_FORWARD_VISIBILITY_DISTANCE], [0, max_dimension * 2]])
    view_segment = affinity.rotate(
        view_segment, -performer_start['rotation']['y'], origin=(0, 0))
    view_segment = affinity.translate(
        view_segment,
        performer_start['position']['x'],
        performer_start['position']['z']
    )
    room_max_x = room_dimensions['x'] / 2.0
    room_max_z = room_dimensions['z'] / 2.0
    room = geometry.box(-room_max_x, -room_max_z, room_max_x, room_max_z)

    target_segment = room.intersection(view_segment)
    if target_segment.is_empty:
        logging.debug(
            f'performer too close to the wall, cannot place object in front '
            f'of it (performer location={performer_start})')
        return None
    return target_segment


def get_location_in_front_of_performer(
    performer_start: Dict[str, Dict[str, float]],
    # TODO MCS-697 Define an ObjectInstance class extending ObjectDefinition.
    target_definition_or_instance: Union[ObjectDefinition, Dict[str, Any]],
    rotation_func: Callable[[], float] = random_rotation,
    room_dimensions: Dict[str, float] = None
) -> Optional[Dict[str, Any]]:
    """Return a random location in the line directly in front of the performer
    agent's starting position and rotation."""

    visible_segment = _get_visible_segment(
        performer_start,
        room_dimensions or DEFAULT_ROOM_DIMENSIONS
    )
    if not visible_segment:
        return None

    def segment_xz(_room_dimensions: Dict[str, float]) -> Tuple[float, float]:
        fraction = random.random()
        point = visible_segment.interpolate(fraction, normalized=True)
        return point.x, point.y

    return calc_obj_pos(
        performer_start['position'],
        [],
        target_definition_or_instance,
        xz_func=segment_xz,
        rotation_func=rotation_func,
        room_dimensions=(room_dimensions or DEFAULT_ROOM_DIMENSIONS)
    )


def get_location_in_back_of_performer(
    performer_start: Dict[str, Dict[str, float]],
    # TODO MCS-697 Define an ObjectInstance class extending ObjectDefinition.
    target_definition_or_instance: Union[ObjectDefinition, Dict[str, Any]],
    rotation_func: Callable[[], float] = random_rotation,
    room_dimensions: Dict[str, float] = None
) -> Optional[Dict[str, Any]]:
    """Return a random location in the 180-degree-arc directly in back of the
    performer agent's starting position and rotation."""

    # TODO MCS-697 Define an ObjectInstance class extending ObjectDefinition.
    if isinstance(target_definition_or_instance, dict):
        debug = target_definition_or_instance['debug']
        dimensions = debug['dimensions']
        offset = debug.get('offset', {'x': 0, 'z': 0})
    else:
        dimensions = vars(target_definition_or_instance.dimensions)
        offset = vars(target_definition_or_instance.offset)

    # We may rotate the object so use its larger X/Z size.
    object_half_size = max(
        dimensions['x'] / 2.0 - offset['x'],
        dimensions['z'] / 2.0 - offset['z']
    )

    # Find the part of the room that's behind the performer's start location
    # (the 180 degree arc in the opposite direction from its start rotation).
    # If the performer would start at (0, 0) facing north (its rotation is 0),
    # its rear poly would then be: minx, miny, maxx, maxy
    rear_poly = geometry.box(
        -((room_dimensions or DEFAULT_ROOM_DIMENSIONS)['x']),
        -((room_dimensions or DEFAULT_ROOM_DIMENSIONS)['z']),
        ((room_dimensions or DEFAULT_ROOM_DIMENSIONS)['x']),
        -0.5 - object_half_size
    )
    # Move the rear poly to behind the performer's start location.
    performer_point = geometry.Point(
        performer_start['position']['x'],
        performer_start['position']['z']
    )
    rear_poly = affinity.translate(
        rear_poly,
        performer_point.x,
        performer_point.y
    )
    # Rotate the rear poly based on the performer's start rotation.
    rear_poly = affinity.rotate(
        rear_poly,
        -performer_start['rotation']['y'],
        origin=performer_point
    )
    # Restrict the rear poly to the room's dimensions.
    room_max_x = (room_dimensions or DEFAULT_ROOM_DIMENSIONS)['x'] / 2.0
    room_max_z = (room_dimensions or DEFAULT_ROOM_DIMENSIONS)['z'] / 2.0
    room_poly = geometry.box(
        -room_max_x + object_half_size,
        -room_max_z + object_half_size,
        room_max_x - object_half_size,
        room_max_z - object_half_size
    )
    rear_poly = rear_poly.intersection(room_poly)

    if not rear_poly or rear_poly.is_empty:
        return None

    rear_min_x = min(room_max_x, max(-room_max_x, rear_poly.bounds[0]))
    rear_max_x = min(room_max_x, max(-room_max_x, rear_poly.bounds[2]))
    rear_min_z = min(room_max_z, max(-room_max_z, rear_poly.bounds[1]))
    rear_max_z = min(room_max_z, max(-room_max_z, rear_poly.bounds[3]))

    def compute_xz(_room_dimensions: Dict[str, float]) -> Tuple[float, float]:
        for _ in range(MAX_TRIES):
            # Try choosing a random X within the rear X bounds.
            x = random_real(rear_min_x, rear_max_x)
            # Draw a vertical line with that X within the rear Z bounds.
            vertical_line = geometry.LineString(
                [[x, rear_min_z], [x, rear_max_z]]
            )
            # Restrict that vertical line to within the full rear bounds.
            line_in_rear = vertical_line.intersection(rear_poly)
            # Try again if the chosen X doesn't work.
            if not line_in_rear.is_empty:
                break
            line_in_rear = None
        # Location is not possible.
        if not line_in_rear or line_in_rear.is_empty:
            return None, None
        # Unlikely but possible to find just a single point here.
        elif line_in_rear.geom_type == 'Point':
            location = line_in_rear
        else:
            # Choose a random position on that vertical line.
            fraction = random.random()
            location = line_in_rear.interpolate(fraction, normalized=True)
        return location.x, location.y

    return calc_obj_pos(
        performer_start['position'],
        [],
        target_definition_or_instance,
        xz_func=compute_xz,
        rotation_func=rotation_func,
        room_dimensions=(room_dimensions or DEFAULT_ROOM_DIMENSIONS)
    )


def get_location_adjacent_to_performer(
    performer_start: Dict[str, Dict[str, float]],
    # TODO MCS-697 Define an ObjectInstance class extending ObjectDefinition.
    target_definition_or_instance: Union[ObjectDefinition, Dict[str, Any]],
    distance: float, direction_rotation: int,
    room_dimensions: Dict[str, float] = None
) -> Optional[Dict[str, Any]]:
    """Returns the instance positioned relative to the performer.  It will be
    placed in a direction based on the `direction_rotation` relative to the
    performers starting rotation.  The `distance` field is the dimensions edge
    to edge distance."""
    if isinstance(target_definition_or_instance, dict):
        debug = target_definition_or_instance['debug']
        dimensions = debug['dimensions']
        offset = debug.get('offset', {'x': 0, 'z': 0})
    else:
        dimensions = vars(target_definition_or_instance.dimensions)
        offset = vars(target_definition_or_instance.offset)

    perf_rot = performer_start['rotation']['y']

    # determine if we want the x orientation facing the performer
    x_oriented = random.choice([True, False])

    # rotation only for the object to match which dimension we use
    extra_obj_rot = (random.choice([90, 270]) if x_oriented
                     else random.choice([0, 180]))

    def compute_xz(_room_dimensions: Dict[str, float]) -> Tuple[float, float]:
        # Desired edge to edge distance

        obj_half_dist = (dimensions['x'] if
                         x_oriented else dimensions['z']) / 2
        origin_dist = PERFORMER_HALF_WIDTH + distance + obj_half_dist

        calc_rot = (perf_rot + direction_rotation)

        x = origin_dist * math.sin(math.radians(calc_rot)) + \
            performer_start['position']['x'] + offset['x']
        z = origin_dist * math.cos(math.radians(calc_rot)) + \
            performer_start['position']['z'] + offset['z']

        return x, z

    def compute_rot():
        return (perf_rot + direction_rotation + extra_obj_rot) % 360

    return calc_obj_pos(
        performer_start['position'],
        [],
        target_definition_or_instance,
        xz_func=compute_xz,
        rotation_func=compute_rot,
        room_dimensions=(room_dimensions or DEFAULT_ROOM_DIMENSIONS)
    )


def get_location_along_wall(
        performer_start: Dict[str, Dict[str, float]],
        wall: str, instance: Dict[str, Any],
        room_dimensions: Dict[str, Any]) -> Dict[str, Any]:
    """returns an object with a new location along a wall for a given room.
    The `wall` parameter must be either `back_wall`, `front_wall`,
    `left_wall`, or `right_wall`"""
    dimensions = instance['debug']['dimensions']
    offset_x = instance['debug']['offset']['x']
    offset_z = instance['debug']['offset']['z']

    def compute_xz(_room_dimensions: Dict[str, float]) -> Tuple[float, float]:
        return get_along_wall_xz(
            wall, _room_dimensions, dimensions, offset_x, offset_z)

    return calc_obj_pos(
        performer_start['position'],
        [],
        instance,
        xz_func=compute_xz,
        room_dimensions=(room_dimensions or DEFAULT_ROOM_DIMENSIONS)
    )


def get_along_wall_xz(wall_label, room_dimensions, dimensions,
                      offset_x=0, offset_z=0):
    """Return an x, z coordinate for a location along the wall.  The
    `wall_label` parameter must be either `back_wall`, `front_wall`,
    `left_wall`, or `right_wall`"""
    buffer = 0.01

    x_limit = room_dimensions['x'] / 2
    z_limit = room_dimensions['z'] / 2

    dim_x = dimensions['x']
    dim_z = dimensions['z']

    # Is this the proper use of offset?
    x_min = -x_limit + dim_x / 2 + offset_x + buffer
    x_max = x_limit - dim_x / 2 - offset_x - buffer
    z_min = -z_limit + dim_z / 2 + offset_z + buffer
    z_max = z_limit - dim_z / 2 - offset_z - buffer

    # Verify front/back term is consistent
    if wall_label == BACK_WALL_LABEL:
        z_min = z_max
    elif wall_label == FRONT_WALL_LABEL:
        z_max = z_min
    elif wall_label == LEFT_WALL_LABEL:
        x_max = x_min
    elif wall_label == RIGHT_WALL_LABEL:
        x_min = x_max
    else:
        raise Exception(f"{wall_label} is not a valid wall label.")

    x = random.uniform(x_min, x_max)
    z = random.uniform(z_min, z_max)
    return x, z


def generate_location_on_object(
    # TODO MCS-697 Define an ObjectInstance class extending ObjectDefinition.
    object_definition_or_instance: Union[ObjectDefinition, Dict[str, Any]],
    # TODO MCS-697 Define an ObjectInstance class extending ObjectDefinition.
    static_instance: Dict[str, Any],
    performer_start: Dict[str, Dict[str, float]],
    bounds_list: List[ObjectBounds],
    room_dimensions: Dict[str, float] = None,
    center: bool = False,
    position_relative_to_start: Dict[str, float] = None
) -> Dict[str, Any]:
    """Creates a location for the object to place it on top of the static object.

    Returns:
        Dict[str, Any]: Location dict or None if it cannot be determined.
    """
    # TODO MCS-697 Define an ObjectInstance class extending ObjectDefinition.
    if isinstance(object_definition_or_instance, dict):
        debug = object_definition_or_instance['debug']
        dimensions = debug['dimensions']
        offset = debug.get('offset', {'x': 0, 'y': 0, 'z': 0})
        position_y = debug.get('positionY', 0)
        rotation = debug.get('rotation', {'x': 0, 'y': 0, 'z': 0})
    else:
        dimensions = vars(object_definition_or_instance.dimensions)
        offset = vars(object_definition_or_instance.offset)
        position_y = object_definition_or_instance.positionY
        rotation = vars(object_definition_or_instance.rotation)

    if not isinstance(static_instance, dict):
        raise Exception(
            "Generate_location_on_object() must be passed a static instance")

    # determine bounds of static object
    static_bounds = (
        static_instance['shows'][0]['boundingBox']
    )
    xy_bounds = static_bounds.box_xz

    location = None
    for _ in range(MAX_TRIES):
        # determine x,z of object in static bounds
        if center:
            x, z = static_bounds.polygon_xz.centroid.coords[0]

            if(position_relative_to_start is not None):
                relative_obj_dim = static_instance['debug']['dimensions']
                buffer = max(dimensions['x'], dimensions['z']) / 2.0

                # determine how much to move away from center/where to place
                # the object
                x_axis_move = relative_obj_dim['x'] / 2.0 - buffer
                z_axis_move = relative_obj_dim['z'] / 2.0 - buffer

                # take rotation of relative object into account
                rel_obj_rot = static_instance['shows'][0]['rotation']['y']

                radians = math.radians(rel_obj_rot)
                new_x = x + (x_axis_move * position_relative_to_start.x)
                new_z = z + (z_axis_move * position_relative_to_start.z)

                # rotation around arbitrary center is:
                # x1 = (x0 -xc) cos(theta) - (z0 -zc)sin(theta) + xc
                # z1 = (x0 -xc) sin(theat) + (z0 -zc)cos(theta) + zc
                #
                # rotate placement point around centroid point of relative
                # object position
                center_x, center_z = x, z
                x = (new_x - center_x) * math.cos(radians) - \
                    (new_z - center_z) * math.sin(radians) + center_x

                z = -(new_x - center_x) * math.sin(radians) - \
                    (new_z - center_z) * math.cos(radians) + center_z
        else:
            # Create some basic bounds to limit where we place the object
            # We will test more precisely later
            buffer = min(dimensions['x'], dimensions['z'])
            xmax = max(xy_bounds[0].x, xy_bounds[1].x,
                       xy_bounds[2].x, xy_bounds[3].x) - buffer
            xmin = min(xy_bounds[0].x, xy_bounds[1].x,
                       xy_bounds[2].x, xy_bounds[3].x) + buffer
            zmax = max(xy_bounds[0].z, xy_bounds[1].z,
                       xy_bounds[2].z, xy_bounds[3].z) - buffer
            zmin = min(xy_bounds[0].z, xy_bounds[1].z,
                       xy_bounds[2].z, xy_bounds[3].z) + buffer
            if xmin > xmax or zmin > zmax:
                raise Exception(
                    "Object with keyword location 'on_top' or "
                    "'on_center' too large to fit on base object.")
            x = random.uniform(xmin, xmax)
            z = random.uniform(zmin, zmax)

        # set y position
        y = static_bounds.max_y + position_y
        if static_bounds.max_y > room_dimensions['y'] - PERFORMER_HEIGHT:
            raise Exception(
                f"Object positioned at y={y} is too high for "
                f"room with height={room_dimensions['y']}")
        obj_position = {'x': x, 'y': y, 'z': z}

        # determine new bounds for object
        object_bounds = create_bounds(
            dimensions=dimensions,
            offset=offset,
            position=obj_position,
            rotation=rotation,
            standing_y=position_y
        )

        # only continue if the object is within the bounds of the static object
        if static_bounds.polygon_xz.contains(object_bounds.polygon_xz):
            location = {
                'position': obj_position,
                'rotation': {
                    'x': rotation['x'],
                    'y': rotation['y'],
                    'z': rotation['z']
                },
                'boundingBox': object_bounds
            }

            if validate_location_rect(
                object_bounds,
                performer_start['position'],
                bounds_list + [static_bounds],
                room_dimensions or DEFAULT_ROOM_DIMENSIONS
            ):
                break
    return location


def generate_location_in_line_with_object(
    # TODO MCS-697 Define an ObjectInstance class extending ObjectDefinition.
    object_definition_or_instance: Union[ObjectDefinition, Dict[str, Any]],
    # TODO MCS-697 Define an ObjectInstance class extending ObjectDefinition.
    static_definition_or_instance: Union[ObjectDefinition, Dict[str, Any]],
    static_location: Dict[str, Any],
    performer_start: Dict[str, Dict[str, float]],
    bounds_list: List[ObjectBounds],
    adjacent: bool = False,
    behind: bool = False,
    obstruct: bool = False,
    unreachable: bool = False,
    room_dimensions: Dict[str, float] = None
) -> Dict[str, Any]:
    """Generate and return a new location for the given object so it's in line
    with the given static object at its given location based on the given
    performer start location. By default, the location is in front of the
    static object. If adjacent=True, the location is to the left or the right
    of the static object. If behind=True, the location is in back of the static
    object. If obstruct=True, the location must obstruct the static object.
    If unreachable=True, the location must be far enough away from the static
    object so that the performer cannot reach the static object. Assume only
    one bool flag is ever used."""

    # TODO MCS-697 Define an ObjectInstance class extending ObjectDefinition.
    if isinstance(object_definition_or_instance, dict):
        debug = object_definition_or_instance['debug']
        dimensions = debug['dimensions']
        offset = debug.get('offset', {'x': 0, 'z': 0})
        position_y = debug.get('positionY', 0)
        rotation = debug.get('rotation', {'x': 0, 'y': 0, 'z': 0})
    else:
        dimensions = vars(object_definition_or_instance.dimensions)
        offset = vars(object_definition_or_instance.offset)
        position_y = object_definition_or_instance.positionY
        rotation = vars(object_definition_or_instance.rotation)

    # TODO MCS-697 Define an ObjectInstance class extending ObjectDefinition.
    if isinstance(static_definition_or_instance, dict):
        static_debug = static_definition_or_instance['debug']
        static_dimensions = static_debug['dimensions']
        static_offset = static_debug.get('offset', {'x': 0, 'y': 0, 'z': 0})
        static_position_y = static_debug.get('positionY', 0)
    else:
        static_dimensions = vars(static_definition_or_instance.dimensions)
        static_offset = vars(static_definition_or_instance.offset)
        static_position_y = static_definition_or_instance.positionY

    static_x = (static_location['position']['x'] + static_offset['x'])
    static_z = (static_location['position']['z'] + static_offset['z'])
    static_bounds = create_bounds(
        dimensions=static_dimensions,
        offset=static_offset,
        position=static_location['position'],
        rotation=static_location['rotation'],
        standing_y=static_position_y
    )

    # The distance needs to be at least the min dimensions of the two objects
    # added together with a gap in between them.
    min_distance_between_objects = MIN_GAP + min(
        (static_dimensions['x'] / 2.0),
        (static_dimensions['z'] / 2.0)
    ) + min((dimensions['x'] / 2.0), (dimensions['z'] / 2.0))

    distance_from_performer_start_to_object = math.sqrt(
        (static_x - performer_start['position']['x'])**2 +
        (static_z - performer_start['position']['z'])**2
    )

    if (not adjacent) and (not behind) and (
        distance_from_performer_start_to_object <
        (min_distance_between_objects * 2)
    ):
        return None

    diagonal_distance_between_objects = MIN_GAP + math.sqrt(
        (static_dimensions['x'] / 2.0)**2 +
        (static_dimensions['z'] / 2.0)**2
    ) + math.sqrt(
        (dimensions['x'] / 2.0)**2 +
        (dimensions['z'] / 2.0)**2
    )

    # The distance shouldn't need to be any more than the diagonal distance
    # between the two objects added together with a gap in between them, unless
    # obstruct=True or unreachable=True, then use the distance between the
    # static object's location and the performer's start location. Subtract the
    # min_distance_between_objects to account for the objects' dimensions.
    max_distance = (
        distance_from_performer_start_to_object - min_distance_between_objects
    ) if (obstruct or unreachable) else diagonal_distance_between_objects

    # Find the angle drawn between the static object and the performer start.
    # Shapely angle: 0 = right, 90 = front, 180 = left, 270 = back
    performer_angle = math.degrees(math.atan2(
        (static_z - performer_start['position']['z']),
        (static_x - performer_start['position']['x'])
    ))
    # If behind=True, rotate the angle by 180 degrees.
    line_rotation_list = [performer_angle] if behind else (
        # If adjacent=True, rotate the angle by 90 or 270 degrees.
        [performer_angle + 90, performer_angle + 270] if adjacent
        else [performer_angle + 180]
    )
    random.shuffle(line_rotation_list)

    # Rotate the object to align with the performer. Needed for obstacle tasks.
    object_rotation = {
        'x': rotation['x'],
        # Transform the Shapely angle into a Unity rotation.
        'y': 450 - performer_angle,
        'z': rotation['z']
    }

    location = None
    for line_rotation in line_rotation_list:
        line_distance = min_distance_between_objects
        # Try to position the 2nd object at a distance away from the static
        # object. If that doesn't work, increase the distance a little bit.
        while line_distance <= max_distance:
            # Create a line of the distance and rotation away from the static
            # object to identify the 2nd object's possible location.
            line = geometry.LineString([[0, 0], [line_distance, 0]])
            line = affinity.rotate(line, line_rotation, origin=(0, 0))
            line = affinity.translate(line, static_x, static_z)
            x = line.coords[1][0] - offset['x']
            z = line.coords[1][1] - offset['z']
            object_bounds = create_bounds(
                dimensions=dimensions,
                offset=offset,
                position={'x': x, 'y': position_y, 'z': z},
                rotation=object_rotation,
                standing_y=position_y
            )
            # Ensure the location is within the room and doesn't overlap with
            # any existing object location or the performer start location.
            successful = validate_location_rect(
                object_bounds,
                performer_start['position'],
                bounds_list + [static_bounds],
                room_dimensions or DEFAULT_ROOM_DIMENSIONS
            )

            if successful and obstruct:
                object_poly = object_bounds.polygon_xz
                static_poly = static_bounds.polygon_xz
                # If obstruct=True, ensure the location is fairly close to the
                # static object and will completely obstruct it from view.
                if object_poly.distance(static_poly) > MAX_REACH_DISTANCE:
                    successful = False
                if successful and not does_fully_obstruct_target(
                    performer_start['position'],
                    static_location,
                    object_poly
                ):
                    successful = False

            if successful and unreachable:
                object_poly = object_bounds.polygon_xz
                static_poly = static_bounds.polygon_xz
                # If unreachable=True, ensure the location is far enough away
                # from the static object so that the performer cannot reach it.
                reachable_distance = object_poly.distance(static_poly) + min(
                    (dimensions['x'] / 2.0),
                    (dimensions['z'] / 2.0)
                )
                if reachable_distance <= MAX_REACH_DISTANCE:
                    successful = False

            if successful:
                location = {
                    'position': {
                        'x': x,
                        'y': position_y,
                        'z': z
                    },
                    'rotation': object_rotation,
                    'boundingBox': object_bounds
                }
                break
            else:
                location = None
                line_distance += MIN_GAP
        if location:
            break

    return location


def retrieve_obstacle_occluder_definition_list(
    # TODO MCS-697 Define an ObjectInstance class extending ObjectDefinition.
    target_definition_or_instance: Union[ObjectDefinition, Dict[str, Any]],
    definition_dataset: DefinitionDataset,
    is_occluder: bool
) -> Optional[Tuple[ObjectDefinition, int]]:
    """Return each object definition from the given list that is both taller
    and wider/deeper that the given target, tupled with a Y rotation amount.
    If wider (x-axis), rotation 0 is returned; if deeper (z-axis), rotation 90
    is returned. If is_occluder is True, each matching definition must be an
    occluder; otherwise, each matching definition must be an obstacle."""

    # TODO MCS-697 Define an ObjectInstance class extending ObjectDefinition.
    if isinstance(target_definition_or_instance, dict):
        debug = target_definition_or_instance['debug']
        target_dimensions = debug['dimensions']
    else:
        target_dimensions = vars(target_definition_or_instance.dimensions)

    object_attr = ('occluder' if is_occluder else 'obstacle')

    def _callback(definition: ImmutableObjectDefinition) -> bool:
        return getattr(definition, object_attr)

    # Filter down the dataset to have only obstacle/occluder definitions.
    filtered_dataset = definition_dataset.filter_on_custom(_callback)

    # Get the groups from the dataset so we can randomize them correctly.
    definition_groups = filtered_dataset.groups()
    group_indexes = list(range(len(definition_groups)))
    inner_indexes = [
        list(range(len(definition_selections)))
        for definition_selections in definition_groups
    ]

    while any([len(indexes) for indexes in inner_indexes]):
        # Choose a random group, then choose a random list within that group.
        group_index = random.choice(group_indexes)
        inner_index = random.choice(inner_indexes[group_index])
        # Remove the chosen inner index from its list.
        inner_indexes[group_index] = [
            i for i in inner_indexes[group_index] if i != inner_index
        ]
        # If there are no more definitions available for a group, remove it.
        if not len(inner_indexes[group_index]):
            group_indexes = [i for i in group_indexes if i != group_index]
        # Choose a random material for the chosen definition.
        definition = random.choice(definition_groups[group_index][inner_index])
        # Identify the dimensions of the chosen obstacle/occluder.
        dimensions = definition.dimensions
        cannot_walk_over = dimensions.y >= PERFORMER_HALF_WIDTH
        cannot_walk_into = definition.mass > PERFORMER_MASS
        # Only need a larger Y dimension if the object is an occluder.
        if cannot_walk_over and cannot_walk_into and (
            not is_occluder or dimensions.y >= target_dimensions['y']
        ):
            if dimensions.x >= target_dimensions['x']:
                return (definition, 0)
            # If the object's Z is bigger than the target's X, rotate it.
            elif dimensions.z >= target_dimensions['x']:
                return (definition, 90)

    return None


def get_bounding_polygon(
    object_or_location: Dict[str, Any]
) -> geometry.Polygon:
    if 'boundingBox' in object_or_location:
        return object_or_location['boundingBox'].polygon_xz
    show = object_or_location['shows'][0]
    if 'boundingBox' in show:
        return show['boundingBox'].polygon_xz


def are_adjacent(obj_a: Dict[str, Any], obj_b: Dict[str, Any],
                 distance: float = MAX_OBJECTS_ADJACENT_DISTANCE) -> bool:
    poly_a = get_bounding_polygon(obj_a)
    poly_b = get_bounding_polygon(obj_b)
    actual_distance = poly_a.distance(poly_b)
    return actual_distance <= distance


def find_performer_bounds(
    performer_position: Dict[str, float]
) -> ObjectBounds:
    return ObjectBounds(
        box_xz=[Vector3d(
            x=performer_position['x'] - PERFORMER_HALF_WIDTH,
            y=0,
            z=performer_position['z'] - PERFORMER_HALF_WIDTH
        ), Vector3d(
            x=performer_position['x'] - PERFORMER_HALF_WIDTH,
            y=0,
            z=performer_position['z'] + PERFORMER_HALF_WIDTH
        ), Vector3d(
            x=performer_position['x'] + PERFORMER_HALF_WIDTH,
            y=0,
            z=performer_position['z'] + PERFORMER_HALF_WIDTH
        ), Vector3d(
            x=performer_position['x'] + PERFORMER_HALF_WIDTH,
            y=0,
            z=performer_position['z'] - PERFORMER_HALF_WIDTH
        )],
        max_y=performer_position['y'] + PERFORMER_HEIGHT,
        min_y=performer_position['y']
    )


def does_fully_obstruct_target(performer_start_position: Dict[str, float],
                               target_or_location: Dict[str, Any],
                               object_poly: geometry.Polygon) -> bool:
    """Returns whether the given object_poly obstructs each line between the
    given performer_start_position and
    all four corners of the given target object or location."""

    return _does_obstruct_target_helper(
        performer_start_position,
        target_or_location,
        object_poly,
        fully=True
    )


def does_partly_obstruct_target(performer_start_position: Dict[str, float],
                                target_or_location: Dict[str, Any],
                                object_poly: geometry.Polygon) -> bool:
    """Returns whether the given object_poly obstructs one line between the
    given performer_start_position and
    the four corners of the given target object or location."""

    return _does_obstruct_target_helper(
        performer_start_position,
        target_or_location,
        object_poly,
        fully=False
    )


def _does_obstruct_target_helper(performer_start_position: Dict[str, float],
                                 target_or_location: Dict[str, Any],
                                 object_poly: geometry.Polygon,
                                 fully: bool = False) -> bool:

    obstructing_points = 0
    performer_start_coordinates = (
        performer_start_position['x'],
        performer_start_position['z']
    )
    bounds = (
        target_or_location.get('boundingBox') or
        target_or_location['shows'][0]['boundingBox']
    )

    target_poly = bounds.polygon_xz
    target_center = target_poly.centroid.coords[0]
    points = bounds.box_xz + \
        [Vector3d(x=target_center[0], y=0, z=target_center[1])]

    if not fully:
        for index, next_point in enumerate(bounds.box_xz):
            previous_point = bounds.box_xz[(index - 1) if (index > 0) else -1]
            line_full = geometry.LineString([
                (previous_point.x, previous_point.z),
                (next_point.x, next_point.z)
            ])
            full_center = line_full.centroid.coords[0]
            center_point = Vector3d(x=full_center[0], y=0, z=full_center[1])
            points.append(center_point)
            for point_1, point_2 in [
                (previous_point, center_point), (center_point, next_point)
            ]:
                line = geometry.LineString([
                    (point_1.x, point_1.z), (point_2.x, point_2.z)
                ])
                line_center = line.centroid.coords[0]
                points.append(
                    Vector3d(x=line_center[0], y=0, z=line_center[1]))

    for point in points:
        target_corner_coordinates = (point.x, point.z)
        line_to_target = geometry.LineString([
            performer_start_coordinates,
            target_corner_coordinates
        ])
        if object_poly.intersects(line_to_target):
            obstructing_points += 1
    return obstructing_points == 5 if fully else obstructing_points > 0


def validate_location_rect(
    location_bounds: ObjectBounds,
    performer_start_position: Dict[str, float],
    bounds_list: List[ObjectBounds],
    room_dimensions: Dict[str, float]
) -> bool:
    """Returns if the given location rect is valid using the given performer
    start position and rect list corresponding to other positioned objects."""
    if not location_bounds.is_within_room(room_dimensions):
        return False
    performer_agent_bounds = find_performer_bounds(performer_start_position)
    for bounds in [performer_agent_bounds] + bounds_list:
        # If one bounds is completely above/below another: no collision.
        if (
            location_bounds.min_y >= bounds.max_y or
            location_bounds.max_y <= bounds.min_y
        ):
            continue
        # If one bounds intersects with another: collision! Invalid.
        if sat_entry(location_bounds.box_xz, bounds.box_xz):
            return False
    return True


def move_to_location(
    object_instance: Dict[str, Any],
    object_location: Dict[str, Any]
) -> Dict[str, Any]:
    """Move the given object to the given location and return the object."""
    location = copy.deepcopy(object_location)
    location['position']['x'] -= object_instance['debug']['offset']['x']
    location['position']['z'] -= object_instance['debug']['offset']['z']
    original_rotation = object_instance['debug'].get('originalRotation', {})
    for axis in ['x', 'y', 'z']:
        location['rotation'][axis] += original_rotation.get(axis, 0)
    object_instance['shows'][0]['position'] = location['position']
    object_instance['shows'][0]['rotation'] = location['rotation']
    object_instance['shows'][0]['boundingBox'] = create_bounds(
        dimensions=object_instance['debug']['dimensions'],
        offset=object_instance['debug']['offset'],
        position=location['position'],
        rotation=location['rotation'],
        standing_y=object_instance['debug']['positionY']
    )
    return object_instance


def generate_floor_area_bounds(
        area_x: float, area_z: float) -> ObjectBounds:
    """Generate and return an ObjectBounds for a floor area (a hole or lava)
    with the given coordinates."""
    # Keeping the buffer at 0.5 caused adjacent holes/lava to 'collide'
    buffer = 0.5 - FLOOR_FEATURE_BOUNDS_BUFFER
    points = [
        Vector3d(x=area_x - buffer, y=0, z=area_z - buffer),
        Vector3d(x=area_x + buffer, y=0, z=area_z - buffer),
        Vector3d(x=area_x + buffer, y=0, z=area_z + buffer),
        Vector3d(x=area_x - buffer, y=0, z=area_z + buffer)
    ]
    # Just use an arbitrarily high number for the max_y.
    return ObjectBounds(box_xz=points, max_y=100, min_y=0)


def find_partition_floor_bounds(room_dim, partition):
    bounds = []
    z_half = room_dim.z / 2.0
    x_half = room_dim.x / 2.0
    if partition.leftHalf:
        x_left_scale = x_half * partition.leftHalf
        left_edge = -x_half + x_left_scale
        points = [
            Vector3d(x=-x_half, y=0, z=-z_half),
            Vector3d(x=left_edge, y=0, z=-z_half),
            Vector3d(x=left_edge, y=0, z=z_half),
            Vector3d(x=-x_half, y=0, z=z_half)
        ]
        bounds.append(ObjectBounds(box_xz=points, max_y=100, min_y=0))
    if partition.rightHalf:
        x_right_scale = x_half * partition.rightHalf
        right_edge = x_half - x_right_scale
        points = [
            Vector3d(x=right_edge, y=0, z=-z_half),
            Vector3d(x=x_half, y=0, z=-z_half),
            Vector3d(x=x_half, y=0, z=z_half),
            Vector3d(x=right_edge, y=0, z=z_half)
        ]
        bounds.append(ObjectBounds(box_xz=points, max_y=100, min_y=0))
    return bounds


def object_x_to_occluder_x(
    object_x: float,
    object_z: float,
    occluder_z: float,
    performer_start_x: float,
    performer_start_z: float
) -> float:
    """Return the X position for an occluder in front of an object at the given
    X/Z position."""
    object_distance_x = object_x - performer_start_x
    object_distance_z = object_z - performer_start_z
    occluder_distance_z = occluder_z - performer_start_z
    # Note: This may need to change if we adjust the camera's field of view.
    offset = (object_distance_x / object_distance_z) * occluder_distance_z
    return round(performer_start_x + offset, 4)


def occluder_x_to_object_x(
    occluder_x: float,
    occluder_z: float,
    object_z: float,
    performer_start_x: float,
    performer_start_z: float
) -> float:
    """Return the X position for an object at the given Z position behind an
    occluder at the given X position."""
    object_distance_z = object_z - performer_start_z
    occluder_distance_x = occluder_x - performer_start_x
    occluder_distance_z = occluder_z - performer_start_z
    # Note: This may need to change if we adjust the camera's field of view.
    offset = (occluder_distance_x / occluder_distance_z) * object_distance_z
    return round(performer_start_x + offset, 4)


def calculate_rotations(
    one: Vector3d,
    two: Vector3d,
    no_rounding_to_tens: bool = False
) -> Tuple[int, int]:
    """Calculates and returns the X and Y rotations for the performer agent in
    position one to look at the object centered in position two. By default,
    will round to nearest 10 (because Rotate actions are in increments of 10)
    unless no_rounding_to_tens is False."""
    dx = -(one.x - two.x)
    dy = one.y + PERFORMER_CAMERA_Y - two.y
    dz = -(one.z - two.z)
    rotation_y = math.degrees(math.atan2(float(dx), dz))
    rotation_x = math.degrees(math.atan2(dy, math.sqrt(dx * dx + dz * dz)))
    if not no_rounding_to_tens:
        rotation_y = round(rotation_y, -1)
        rotation_x = round(rotation_x, -1)
    return int(round(rotation_x)), int(round(rotation_y)) % 360


def get_position_distance_away_from_obj(
        room_dimensions: Vector3d, obj,
        distance, bounds) -> Tuple[float, float]:
    """ Calculates and returns a position around an object at a
    specified distance away from an edge of the objects bounding box
    """
    bb_boxes = obj['shows'][0]['boundingBox'].box_xz
    top_right = geometry.Point(bb_boxes[0].x, bb_boxes[0].z)
    bottom_right = geometry.Point(bb_boxes[1].x, bb_boxes[1].z)
    bottom_left = geometry.Point(bb_boxes[2].x, bb_boxes[2].z)
    top_left = geometry.Point(bb_boxes[3].x, bb_boxes[3].z)
    normalized_vertical_vector = \
        get_normalized_vector_from_two_points(
            top_right, bottom_right)
    normalized_horizontal_vector = \
        get_normalized_vector_from_two_points(
            top_right, top_left)
    distance_away = distance + PERFORMER_HALF_WIDTH
    """
    p4----------p1
    |  tl   tr  |
    |    [o]    |
    |    [b]    |
    |    [j]    |
    |  bl   br  |
    p3---------p2
    """
    resultant_vector_up_right = Vector3d(
        x=(normalized_vertical_vector.x +
           normalized_horizontal_vector.x) * distance_away,
        z=(normalized_vertical_vector.y +
           normalized_horizontal_vector.y) * distance_away)
    resultant_vector_down_right = Vector3d(
        x=(-normalized_vertical_vector.x +
           normalized_horizontal_vector.x) * distance_away,
        z=(-normalized_vertical_vector.y +
            normalized_horizontal_vector.y) * distance_away)
    resultant_vector_down_left = Vector3d(
        x=-(normalized_vertical_vector.x +
            normalized_horizontal_vector.x) * distance_away,
        z=-(normalized_vertical_vector.y +
            normalized_horizontal_vector.y) * distance_away)
    resultant_vector_up_left = Vector3d(
        x=(normalized_vertical_vector.x -
           normalized_horizontal_vector.x) * distance_away,
        z=(normalized_vertical_vector.y -
            normalized_horizontal_vector.y) * distance_away)

    p1 = geometry.Point(
        top_right.x + resultant_vector_up_right.x,
        top_right.y + resultant_vector_up_right.z)
    p2 = geometry.Point(
        bottom_right.x + resultant_vector_down_right.x,
        bottom_right.y + resultant_vector_down_right.z)
    p3 = geometry.Point(
        bottom_left.x + resultant_vector_down_left.x,
        bottom_left.y + resultant_vector_down_left.z)
    p4 = geometry.Point(
        top_left.x + resultant_vector_up_left.x,
        top_left.y + resultant_vector_up_left.z)

    pos = Vector3d(x=0, y=0, z=0)
    # The perimeter line
    line = geometry.LineString((
        [p1.x, p1.y],
        [p2.x, p2.y],
        [p3.x, p3.y],
        [p4.x, p4.y],
        [p1.x, p1.y]))
    mp = geometry.MultiPoint()
    # Seperation of spawn points on the line
    seperation_between_spawn_points = 0.5
    for i in np.arange(0, line.length,
                       seperation_between_spawn_points):
        segment = ops.substring(
            line, i, i + seperation_between_spawn_points)
        mp = mp.union(segment.boundary)
    x = [point.x for point in mp]
    y = [point.y for point in mp]
    for _ in range(MAX_TRIES):
        index = random.randint(0, len(x) - 1)
        pos.x = x[index]
        pos.z = y[index]
        """
        If you want to visually see how this picks points around the
        perimeter uncomment this
        import matplotlib.pyplot as plt
        plt.gca().set_aspect('equal', adjustable='box')
        plt.scatter(x, y)
        x_point = [pos.x]
        y_point = [pos.z]
        plt.scatter(x_point, y_point)
        plt.savefig("perimeter_graph.png")
        """
        """
        Check the new performer start bounds.
        Move the old performer start out of the room for the check.
        """
        performer_bounds = find_performer_bounds(vars(pos))
        if not performer_bounds.is_within_room(vars(room_dimensions)):
            continue
        valid = validate_location_rect(
            location_bounds=performer_bounds,
            performer_start_position=vars(
                Vector3d(x=math.inf, y=math.inf, z=math.inf)),
            bounds_list=bounds,
            room_dimensions=vars(room_dimensions)
        )
        if not valid:
            continue

        """
        This is another check to verify the distance is correct
        If its not, it means the position is in the corner of the
        perimeter. Try to nudge the position a little bit in a direction.
        If that nudge moves it further away, go a different direction.
        Once the direction is found, then keep nudging it from the corner
        toward the object until the distance is correct. This is kind of like
        turning a corner into an oval.
        """
        (valid, start_difference) = \
            distance_between_point_and_bounding_box_is_valid(
                pos.x, pos.z, top_right, bottom_right,
                bottom_left, top_left, distance_away)
        if not valid:
            directions = [
                resultant_vector_up_left, resultant_vector_down_right,
                resultant_vector_down_left, resultant_vector_up_left]
            for direction in directions:
                shift_amount = start_difference
                shift_amount = start_difference
                for _ in range(MAX_TRIES):
                    reverse_x = -direction.x * shift_amount
                    reverse_z = -direction.z * shift_amount
                    new_pos_x = pos.x + reverse_x
                    new_pos_z = pos.z + reverse_z
                    (valid, difference_after_shift) = \
                        distance_between_point_and_bounding_box_is_valid(
                            new_pos_x, new_pos_z,
                            top_right, bottom_right, bottom_left,
                            top_left, distance_away)
                    shift_amount += 0.01
                    # the shift is in the wrong direction
                    if difference_after_shift > start_difference:
                        break
                    if valid:
                        return (new_pos_x, new_pos_z)
        if valid:
            return (pos.x, pos.z)

    raise Exception(
        f"Failed to find valid performer location "
        f"with distance away: ({distance}) from object: ({obj['id']})"
        f"because location is obstructed or outside of room bounds")


def distance_between_point_and_bounding_box_is_valid(
        x, z, top_right, bottom_right,
        bottom_left, top_left, target_distance_away):
    """
    Checks if an x,z position is a distance away from a bounding box.
    """
    point = geometry.Point(x, z)
    polygon = geometry.Polygon(
        [top_right, bottom_right, bottom_left, top_left])
    target_distance_away = round(target_distance_away, 2)
    end_distance = round(point.distance(polygon), 2)
    difference = round(end_distance - target_distance_away, 2)
    valid = end_distance == target_distance_away
    return valid, difference


def get_normalized_vector_from_two_points(
        p1: geometry.Point, p2: geometry.Point):
    """Calculates and returns a normalized vector between two points"""
    # got this math from here https://math.stackexchange.com/a/175906
    vector_direction = Vector3d(x=p1.x - p2.x, y=p1.y - p2.y)
    length_of_vector = math.sqrt(
        vector_direction.x ** 2 +
        vector_direction.y ** 2)
    normalized_vector = Vector3d(
        x=vector_direction.x / length_of_vector,
        y=vector_direction.y / length_of_vector)
    return normalized_vector
