import os

import pytest
import shapely
from shapely import affinity

from generator import RetrievalGoal, definitions, geometry, scene_saver, tags
from hypercube import (
    InteractiveContainerEvaluation4HypercubeFactory,
    InteractiveObstacleEvaluationHypercubeFactory,
    InteractiveOccluderEvaluationHypercubeFactory,
    InteractiveSingleSceneFactory
)
from tests.interactive_hypercube_test_util import (
    STARTER_SCENE,
    map_id_to_scene,
    verify_scene
)


class ExpectedLocation():
    def __init__(self, target_data, object_data_list, receptacle_data=None):
        self.data = {}

        for object_data in object_data_list:
            for index in range(len(object_data.instance_list)):
                self.set(object_data, target_data, index, receptacle_data)

    def get(
        self,
        object_id,
        location_name,
        target_location_name=None,
        is_untrained=False,
        is_receptacle_untrained=False
    ):
        return self.data[self.key(
            object_id,
            location_name,
            target_location_name,
            is_untrained,
            is_receptacle_untrained
        )]

    def key(
        self,
        object_id,
        location_name,
        target_location_name=None,
        is_untrained=False,
        is_receptacle_untrained=False
    ):
        return (
            object_id + '_' + ('untrained' if is_untrained else 'trained') +
            '_' + location_name +
            (('_' + target_location_name) if target_location_name else '') +
            (
                ('_' + ('untrained' if is_receptacle_untrained else 'trained'))
                if location_name.lower().startswith('inside') else ''
            )
        ).lower()

    def set(self, object_data, target_data, index, receptacle_data=None):
        instance = object_data.instance_list[index]
        if not instance:
            return
        key = self.key(
            instance['id'],
            object_data.location_plan_list[index].name,
            target_data.location_plan_list[index].name
            if instance['debug']['role'] != 'target' else None,
            object_data.untrained_plan_list[index],
            receptacle_data.untrained_plan_list[index] if receptacle_data
            else False
        )
        if key not in self.data:
            self.data[key] = get_position(instance)


def get_position(instance):
    return instance['shows'][0]['position']


def get_object(object_id, object_list, optional=False):
    filtered_list = [
        instance for instance in object_list
        if instance['id'] == object_id
    ]
    if optional:
        assert (len(filtered_list) == 1 or len(filtered_list) == 0)
    else:
        assert len(filtered_list) == 1
    return filtered_list[0] if len(filtered_list) > 0 else None


def delete_scene_debug_files(scene_dict, base_filename):
    for scene_id in scene_dict.keys():
        os.remove(f'{base_filename}_{scene_id.upper()}_debug.json')


def save_scene_debug_files(scene_dict, descriptor):
    base_filename, _ = scene_saver.find_next_filename(
        f'temp_{descriptor}_',
        1,
        '04',
        suffix='_A1_debug.json'
    )
    for scene_id, scene_data in scene_dict.items():
        scene, _ = scene_data
        scene_saver.save_scene_files(
            scene,
            base_filename,
            only_debug_file=True
        )
    return base_filename


def get_parent(object_instance, object_list):
    parent_id = object_instance['locationParent']
    parent = next((o for o in object_list if o['id'] == parent_id))
    return parent


def verify_confusor(target, confusor):
    assert target['id'] != confusor['id']
    is_same_color = (
        set(target['debug']['color']) == set(confusor['debug']['color'])
    )
    is_same_shape = target['debug']['shape'] == confusor['debug']['shape']
    is_same_size = target['debug']['size'] == confusor['debug']['size'] and \
        (target['debug']['dimensions']['x'] - definitions.MAX_SIZE_DIFF) <= \
        confusor['debug']['dimensions']['x'] <= \
        (target['debug']['dimensions']['x'] + definitions.MAX_SIZE_DIFF) and \
        (target['debug']['dimensions']['y'] - definitions.MAX_SIZE_DIFF) <= \
        confusor['debug']['dimensions']['y'] <= \
        (target['debug']['dimensions']['y'] + definitions.MAX_SIZE_DIFF) and \
        (target['debug']['dimensions']['z'] - definitions.MAX_SIZE_DIFF) <= \
        confusor['debug']['dimensions']['z'] <= \
        (target['debug']['dimensions']['z'] + definitions.MAX_SIZE_DIFF)
    if (
        (not is_same_color and not is_same_shape) or
        (not is_same_color and not is_same_size) or
        (not is_same_shape and not is_same_size) or
        (is_same_color and is_same_shape and is_same_size)
    ):
        print(
            f'[ERROR] CONFUSOR SHOULD BE THE SAME AS THE TARGET IN ALL '
            f'EXCEPT ONE: color={is_same_color} shape={is_same_shape} '
            f'size={is_same_size}\ntarget={target}\nconfusor={confusor}')
        return False
    return True


def verify_immediately_visible(
    performer_start,
    object_list,
    target,
    adjacent=None,
    hide_errors=False
):
    """Verify the given target is immediately visible using the given performer
    start location (excluding the given adjacent object, if any)."""

    target_or_parent = get_parent(target, object_list) \
        if 'locationParent' in target else target
    target_poly = geometry.get_bounding_polygon(target_or_parent)

    view_line = shapely.geometry.LineString([[0, 0], [0, 10]])
    view_line = affinity.rotate(
        view_line,
        -performer_start['rotation']['y'],
        origin=(0, 0)
    )
    view_line = affinity.translate(
        view_line,
        performer_start['position']['x'],
        performer_start['position']['z']
    )

    target_role = target['debug']['role']
    if not target_poly.intersection(view_line):
        if not hide_errors:
            print(f'[ERROR] {target_role.upper()} SHOULD BE VISIBLE IN '
                  f'FRONT OF PERFORMER:\n{target_role}={target_or_parent}\n'
                  f'performer_start={performer_start}')
        return False

    ignore_id_list = [target['id'], target_or_parent['id']]
    if adjacent:
        ignore_id_list.append(adjacent['id'])
        if 'locationParent' in adjacent:
            ignore_id_list.append(get_parent(adjacent)['id'])

    for instance in object_list:
        if (
            instance['id'] not in ignore_id_list and
            'locationParent' not in instance
        ):
            object_poly = geometry.get_bounding_polygon(instance)
            if geometry.does_fully_obstruct_target(
                performer_start['position'],
                target_or_parent,
                object_poly
            ):
                if not hide_errors:
                    instance_role = instance['debug']['role']
                    print(
                        f'[ERROR] {instance_role.upper()} SHOULD NOT '
                        f'OBSTRUCT {target_role.upper()}:\n'
                        f'{target_role}={target_or_parent}\n'
                        f'{instance_role}={instance}\n'
                        f'performer_start={performer_start}'
                    )
                return False

    return True


def verify_not_immediately_visible(
    performer_start,
    object_list,
    target,
    adjacent=None
):
    """Verify the given target isn't immediately visible using the given
    performer start location (excluding the given adjacent object, if any)."""

    result = verify_immediately_visible(
        performer_start,
        object_list,
        target,
        adjacent,
        hide_errors=True
    )
    if result:
        target_or_parent = get_parent(target, object_list) \
            if 'locationParent' in target else target
        target_role = target['debug']['role']
        print(f'[ERROR] {target_role.upper()} SHOULD NOT BE VISIBLE IN '
              f'FRONT OF PERFORMER:\n{target_role}={target_or_parent}\n'
              f'performer_start={performer_start}')
    return not result


def verify_location_obstruct_target(obstructor, target, performer_start):
    """Verify the location of the given (possible) obstructor using the given
    target and performer start location."""

    if not geometry.does_partly_obstruct_target(
        performer_start['position'],
        target,
        geometry.get_bounding_polygon(obstructor)
    ):
        print(
            f'[ERROR] OBJECT SHOULD OBSTRUCT TARGET:\ntarget={target}\n'
            f'object={obstructor}\nperformer_start={performer_start}')
        return False

    return True


def verify_location_not_obstruct_target(obstructor, target, performer_start):
    """Verify the location of the given (possible) obstructor using the given
    target and performer start location."""

    if geometry.does_fully_obstruct_target(
        performer_start['position'],
        target,
        geometry.get_bounding_polygon(obstructor)
    ):
        return False

    return True


def verify_not_parent(
    expected_parent,
    actual_parent,
    expected_target=None,
    actual_target=None
):
    if expected_target and actual_target and not (
        expected_target.get('locationParent', None) is None and
        actual_target.get('locationParent', None) is None
    ):
        print(f'[ERROR] TARGET SHOULD NOT HAVE LOCATION PARENT:\n'
              f'{expected_target}\n{actual_target}')
        return False

    if not (
        expected_parent['debug'].get('isParentOf', []) == [] and
        actual_parent['debug'].get('isParentOf', []) == []
    ):
        print(f'[ERROR] OBJECT SHOULD NOT HAVE TARGET CHILD:\n'
              f'{expected_parent}\n{actual_parent}')
        return False

    return True


def verify_obstacle(obstacle, target, object_list, performer_start):
    """Verify the mass/size of the given obstacle using the given target."""

    print(f'[DEBUG] obstacle={obstacle}')

    dimensions = obstacle['debug']['dimensions']

    if not (
        target['debug']['dimensions']['x'] <= dimensions['x'] or
        target['debug']['dimensions']['z'] <= dimensions['z']
    ):
        print(
            f'[ERROR] OBSTACLE SHOULD BE BIGGER THAN TARGET:\n'
            f'target={target}\nobstacle={obstacle}')
        return False

    if dimensions['y'] < geometry.PERFORMER_HALF_WIDTH:
        print(
            f'[ERROR] OBSTACLE SHOULD BE TALL ENOUGH SO THE PERFORMER CANNOT '
            f'SIMPLY WALK OVER IT:\ntarget={target}\nobstacle={obstacle}')
        return False

    if obstacle['mass'] <= 2:
        print(
            f'[ERROR] OBSTACLE SHOULD BE HEAVY ENOUGH SO THE PERFORMER CANNOT '
            f'SIMPLY MOVE INTO IT AND PUSH IT AWAY:\ntarget={target}'
            f'\nobstacle={obstacle}')
        return False

    for instance in object_list:
        if (
            instance['id'] == obstacle['id'] or
            instance['debug']['role'] in ['target', 'occluder', 'obstacle']
        ):
            continue
        if not verify_location_not_obstruct_target(
            instance,
            obstacle,
            performer_start
        ):
            print(f'[ERROR] OBJECT SHOULD NOT COMPLETELY OBSTRUCT OBSTACLE\n'
                  f'object={instance}\nobstacle={obstacle}\n'
                  f'performer_start={performer_start}')
            return False

    return True


def verify_occluder(occluder, target, object_list, performer_start):
    """Verify the mass/size of the given occluder using the given target."""

    print(f'[DEBUG] occluder={occluder}')

    dimensions = occluder['debug']['dimensions']

    if not (
        target['debug']['dimensions']['x'] <= dimensions['x'] or
        target['debug']['dimensions']['y'] <= dimensions['y'] or
        target['debug']['dimensions']['z'] <= dimensions['z']
    ):
        print(
            f'[ERROR] OCCLUDER SHOULD BE BIGGER THAN TARGET:\n'
            f'target={target}\noccluder={occluder}')
        return False

    if dimensions['y'] < geometry.PERFORMER_HALF_WIDTH:
        print(
            f'[ERROR] OCCLUDER SHOULD BE TALL ENOUGH SO THE PERFORMER CANNOT '
            f'SIMPLY WALK OVER IT:\ntarget={target}\noccluder={occluder}')
        return False

    if occluder['mass'] <= 2:
        print(
            f'[ERROR] OCCLUDER SHOULD BE HEAVY ENOUGH SO THE PERFORMER CANNOT '
            f'SIMPLY MOVE INTO IT AND PUSH IT AWAY:\ntarget={target}'
            f'\noccluder={occluder}')
        return False

    for instance in object_list:
        if (
            instance['id'] == occluder['id'] or
            instance['debug']['role'] in ['target', 'occluder', 'obstacle']
        ):
            continue
        if not verify_location_not_obstruct_target(
            instance,
            occluder,
            performer_start
        ):
            print(f'[ERROR] OBJECT SHOULD NOT COMPLETELY OBSTRUCT OCCLUDER\n'
                  f'object={instance}\noccluder={occluder}\n'
                  f'performer_start={performer_start}')
            return False

    return True


def verify_object_list(scene, index, tag, object_list):
    """Verify each given object is in the scene, and are same as their
    templates. Also verify the goal tags of the scene."""

    for expected in object_list:
        actual = get_object(expected['id'], scene.objects, True)

        assert expected['debug']['role'] == tag
        assert actual['debug']['role'] == tag

        # Verify instance in scene is same as object data instance.
        if not verify_same_object(tag, expected, actual, []):
            print(f'[ERROR] ACTUAL INSTANCE FROM SCENE AT INDEX {index} IS '
                  f'NOT SAME AS EXPECTED INSTANCE FROM OBJECT DATA')
            return False

        assert verify_target_soccer_ball(tag, expected)

        # Verify object instance info is in goal info.
        for info in expected['debug']['info']:
            if info not in scene.goal['objectsInfo']['all']:
                return False
            if info != tag:
                if info not in scene.goal['objectsInfo'][tag]:
                    return False

    if scene.goal['sceneInfo']['count'][tag] != len(object_list):
        return False
    if scene.goal['sceneInfo']['present'][tag] != (len(object_list) > 0):
        return False
    return True


def verify_object_data_list(scene, index, tag, object_data_list, ignore_list):
    """Verify each given object is in the scene, and are same as their
    templates. Also verify the goal tags of the scene."""

    count = 0
    for object_data in object_data_list:
        trained = object_data.trained_template
        untrained = object_data.untrained_template
        expected = object_data.instance_list[index]
        actual = get_object(trained['id'], scene.objects, True)

        if not expected:
            assert not actual
            continue

        count += 1

        assert expected['debug']['role'] == tag
        assert actual['debug']['role'] == tag

        # Verify object data instance is same as template.
        if object_data.untrained_plan_list[index]:
            if not verify_same_object(tag, untrained, expected, ignore_list):
                print(f'[ERROR] INSTANCE FROM OBJECT DATA INDEX {index} IS '
                      f'NOT SAME AS UNTRAINED TEMPLATE')
                return False
        else:
            if not verify_same_object(tag, trained, expected, ignore_list):
                print(f'[ERROR] INSTANCE FROM OBJECT DATA INDEX {index} IS '
                      f'NOT SAME AS TRAINED TEMPLATE')
                return False

        # Verify instance in scene is same as object data instance.
        if not verify_same_object(tag, expected, actual, []):
            print(f'[ERROR] ACTUAL INSTANCE FROM SCENE AT INDEX {index} IS '
                  f'NOT SAME AS EXPECTED INSTANCE FROM OBJECT DATA')
            return False

        assert verify_target_soccer_ball(tag, expected)

        # Verify object instance info is in goal info.
        for info in expected['debug']['info']:
            if info not in scene.goal['objectsInfo']['all']:
                return False
            if info != tag:
                if info not in scene.goal['objectsInfo'][tag]:
                    return False

    if scene.goal['sceneInfo']['count'][tag] != count:
        return False
    if scene.goal['sceneInfo']['present'][tag] != (count > 0):
        return False
    return True


def verify_same_location(role, base, expected, actual):
    if base != expected or expected != actual:
        print(f'[ERROR] {role.upper()} SHOULD HAVE SAME LOCATION:\n'
              f'{base}\n{expected}\n{actual}')
        return False
    return True


def verify_same_object(name, object_1, object_2, ignore_list):
    """Verify the two given objects have all the same properties, except for
    the specific given ignore_list."""

    ignore_list = ignore_list + ['debug', 'info', 'role']
    key_set_1 = set(object_1.keys())
    key_set_2 = set(object_2.keys())
    debug_key_set_1 = set(object_1['debug'].keys())
    debug_key_set_2 = set(object_2['debug'].keys())
    for key in key_set_1.union(key_set_2):
        if key not in ignore_list and not key.startswith('is'):
            if (
                key not in object_1 or
                key not in object_2 or
                object_1[key] != object_2[key]
            ):
                print(
                    f'[ERROR] "{key}" PROPERTY SHOULD BE THE SAME:\n'
                    f'{name}_1={object_1}\n{name}_2={object_2}')
                return False
    for key in debug_key_set_1.union(debug_key_set_2):
        if key not in ignore_list and not key.startswith('is'):
            if (
                key not in object_1['debug'] or
                key not in object_2['debug'] or
                object_1['debug'][key] != object_2['debug'][key]
            ):
                print(
                    f'[ERROR] "{key}" DEBUG PROPERTY SHOULD BE THE SAME:\n'
                    f'{name}_1={object_1}\n{name}_2={object_2}')
                return False
    return True


def verify_same_parent(
    expected_parent,
    actual_parent,
    expected_target,
    actual_target
):
    # Expected and actual IDs are the same (will have been verified before).
    target_id = actual_target['id']
    parent_id = actual_parent['id']

    if not (
        expected_target.get('locationParent') == parent_id and
        actual_target.get('locationParent') == parent_id
    ):
        print(f'[ERROR] TARGET SHOULD HAVE SAME LOCATION PARENT {parent_id} '
              f'BUT ACTUALLY HAD {expected_parent.get("locationParent")} AND '
              f'{actual_parent.get("locationParent")}:\n'
              f'{expected_target}\n{actual_target}')
        return False

    if not (
        expected_parent['debug'].get('isParentOf') == [target_id] and
        actual_parent['debug'].get('isParentOf') == [target_id]
    ):
        print(f'[ERROR] OBJECT SHOULD HAVE SAME TARGET CHILD {target_id} BUT '
              f'ACTUALLY HAD {expected_parent["debug"].get("isParentOf")} AND '
              f'{actual_parent["debug"].get("isParentOf")}:\n'
              f'{expected_parent}\n{actual_parent}')
        return False

    parent_poly = geometry.get_bounding_polygon(actual_parent)
    target_poly = geometry.get_bounding_polygon(actual_target)

    # Adjust the target bounds by the position and rotation of the parent.
    parent_center = get_position(actual_parent)
    target_poly = affinity.translate(
        target_poly,
        parent_center['x'],
        parent_center['z']
    )
    target_poly = affinity.rotate(
        target_poly,
        -actual_parent['shows'][0]['rotation']['y']
    )

    if not parent_poly.contains(target_poly):
        print(f'[ERROR] PARENT BOUNDING BOX SHOULD CONTAIN CHILD:\n'
              f'parent_poly={parent_poly}\nchild_poly={target_poly}\n'
              f'parent={actual_parent}\nchild={actual_target}')
        return False

    return True


def verify_scene_properties(
    hypercube,
    scene,
    index,
    ignore_parent=False,
    ignore_container_location=False,
    ignore_obstacle_location=False,
    ignore_occluder_location=False
):
    """Verify the given scene is valid."""
    task = hypercube.get_name()
    task = task[task.rfind(' ') + 1:]
    slice_count = 4
    if task == 'obstacle':
        slice_count = 3
    if task == 'retrieval':
        # This is just a training scene
        slice_count = 0
    verify_scene(scene, task, slice_count)

    assert 10 <= scene.room_dimensions.x <= 15
    assert 3 <= scene.room_dimensions.y <= 5
    assert 10 <= scene.room_dimensions.z <= 15

    # Floor should not have distracting patterns.
    assert len(scene.debug['floorColors']) == 1

    # Floor should not have the same color as critical (non-context) objects.
    for role in [
        'target', 'confusor', 'large_container', 'small_container',
        'obstacle', 'occluder'
    ]:
        for object_data in hypercube._data[role]:
            if object_data.instance_list[index]:
                for color in (
                    object_data.instance_list[index]['debug']['color']
                ):
                    if color in scene.debug['floorColors']:
                        print(f'[ERROR] FLOOR SHOULD NOT HAVE SAME COLOR AS '
                              f'OBJECT: floor_colors='
                              f'{scene.debug["floorColors"]}\n'
                              f'object={object_data.instance_list[index]}')
                        return False

    # Ensure the performer start in each scene is the same.
    performer_start = {
        'position': vars(scene.performer_start.position),
        'rotation': vars(scene.performer_start.rotation)
    }
    if performer_start != hypercube._performer_start:
        print(
            f'[ERROR] performer_start SHOULD BE THE SAME: '
            f'{performer_start} != {hypercube._performer_start}')
        return False
    performer_start_poly = geometry.find_performer_bounds(
        hypercube._performer_start['position']
    ).polygon_xz
    for instance in scene.objects:
        if instance.get('locationParent'):
            continue
        object_poly = geometry.get_bounding_polygon(instance)
        if object_poly.intersects(performer_start_poly):
            print(
                f'[ERROR] performer_start SHOULD NOT BE INSIDE AN '
                f'OBJECT:\nperformer_start_poly={performer_start_poly}\n'
                f'object_poly={object_poly}\ninstance={instance}')
            return False

    # Expected differences across scenes in objects that may have/be parents.
    parent_ignore_list = (
        ['isParentOf', 'locationParent', 'parentArea', 'canContainTarget']
        if ignore_parent else []
    )

    # Assume that each scene must have the target.
    assert hypercube._target_data.instance_list[index]

    # Assume that target is never untrained.
    assert not hypercube._target_data.untrained_plan_list[index]

    assert verify_object_data_list(
        scene,
        index,
        'target',
        [hypercube._target_data],
        # Expect the object's show location will change across some scenes.
        ['shows'] + parent_ignore_list
    )

    if (
        hypercube._confusor_data and
        hypercube._confusor_data.instance_list[index]
    ):
        assert verify_confusor(
            hypercube._target_data.instance_list[index],
            hypercube._confusor_data.instance_list[index]
        )
        assert verify_object_data_list(
            scene,
            index,
            'confusor',
            [hypercube._confusor_data],
            # Expect the object's show location will change across some scenes.
            ['shows'] + parent_ignore_list
        )

    assert verify_object_data_list(
        scene,
        index,
        'container',
        hypercube._data['large_container'] +
        hypercube._data['small_container'],
        (['shows'] if ignore_container_location else []) + parent_ignore_list
    )

    assert verify_object_list(
        scene,
        index,
        'context',
        hypercube._small_context_object_list
    )

    assert verify_object_data_list(
        scene,
        index,
        'obstacle',
        hypercube._data['obstacle'],
        (['shows'] if ignore_obstacle_location else [])
    )

    performer_start = {
        'position': vars(scene.performer_start.position),
        'rotation': vars(scene.performer_start.rotation)
    }

    for obstacle_data in hypercube._data['obstacle']:
        if obstacle_data.instance_list[index]:
            assert verify_obstacle(
                obstacle_data.instance_list[index],
                hypercube._target_data.instance_list[index],
                scene.objects,
                performer_start
            )

    assert verify_object_data_list(
        scene,
        index,
        'occluder',
        hypercube._data['occluder'],
        (['shows'] if ignore_occluder_location else [])
    )

    for occluder_data in hypercube._data['occluder']:
        if occluder_data.instance_list[index]:
            assert verify_occluder(
                occluder_data.instance_list[index],
                hypercube._target_data.instance_list[index],
                scene.objects,
                performer_start
            )

    assert scene.goal['category'] == tags.tag_to_label(tags.SCENE.RETRIEVAL)
    assert scene.goal['description']
    assert scene.goal['last_step'] >= 2500
    assert scene.goal['sceneInfo']['id']
    assert 'slices' in scene.goal['sceneInfo']

    target_metadata = scene.goal['metadata'].get(
        'target',
        scene.goal['metadata'].get('target_1')
    )

    assert target_metadata['id'] == (
        hypercube._target_data.instance_list[index]['id']
    )
    assert target_metadata['info'] == (
        hypercube._target_data.instance_list[index]['debug']['info']
    )

    return True


def verify_target_obstruct_location(target, obstructor, performer_start):
    """Verify the location of the given (possible) obstructor using the given
    target and performer start location."""

    if not geometry.does_partly_obstruct_target(
        performer_start['position'],
        obstructor,
        geometry.get_bounding_polygon(target)
    ):
        print(
            f'[ERROR] TARGET SHOULD OBSTRUCT OBJECT:\ntarget={target}\n'
            f'object={obstructor}\nperformer_start={performer_start}')
        return False

    return True


def verify_target_soccer_ball(tag, instance):
    if tag == 'target':
        # Eval 4+: Target is always soccer ball with scale 1.
        if (
            instance['type'] != 'soccer_ball' or
            instance['shows'][0]['scale'] != {'x': 1, 'y': 1, 'z': 1}
        ):
            print(f'[ERROR] EXPECTED SOCCER BALL WITH SCALE 1:\n{instance}')
            return False
    else:
        if instance['type'] == 'soccer_ball':
            print(
                f'[ERROR] EXPECTED {tag.upper()} NOT SOCCER BALL:\n{instance}'
            )
            return False
    return True


def verify_target_trophy(tag, instance):
    if tag == 'target':
        # Eval 3: Target is always trophy with scale 1.
        if (
            instance['type'] != 'trophy' or
            instance['shows'][0]['scale'] != {'x': 1, 'y': 1, 'z': 1}
        ):
            print(f'[ERROR] EXPECTED SOCCER BALL WITH SCALE 1:\n{instance}')
            return False
    else:
        if instance['type'] == 'trophy':
            print(
                f'[ERROR] EXPECTED {tag.upper()} NOT SOCCER BALL:\n{instance}'
            )
            return False
    return True


def test_single_scene():
    hypercube_factory = InteractiveSingleSceneFactory(RetrievalGoal(''))
    hypercube = hypercube_factory._build(STARTER_SCENE)

    assert len(hypercube._data['large_container']) == 0
    assert len(hypercube._data['small_container']) == 0
    assert len(hypercube._data['obstacle']) == 0
    assert len(hypercube._data['occluder']) == 0

    assert 0 <= len(hypercube._small_context_object_list) <= 10

    scenes = hypercube.get_scenes()
    assert len(scenes) == 1
    assert verify_scene_properties(hypercube, scenes[0], 0)


def test_eval_4_container_hypercube():
    hypercube_factory = InteractiveContainerEvaluation4HypercubeFactory(
        RetrievalGoal('container')
    )
    hypercube = hypercube_factory._build(STARTER_SCENE)

    assert len(hypercube._data['large_container']) == 3
    assert len(hypercube._data['small_container']) == 0

    assert len(hypercube._data['obstacle']) == 0
    assert len(hypercube._data['occluder']) == 0

    assert 0 <= len(hypercube._small_context_object_list) <= 10

    target_data = hypercube._target_data
    target_id = target_data.trained_template['id']
    large_container_data_1 = hypercube._data['large_container'][0]
    large_container_data_2 = hypercube._data['large_container'][1]
    large_container_data_3 = hypercube._data['large_container'][2]
    large_container_id_1 = large_container_data_1.trained_template['id']
    large_container_id_2 = large_container_data_2.trained_template['id']
    large_container_id_3 = large_container_data_3.trained_template['id']

    # Save a location of each variable-location object, to verify that the
    # objects will always keep the same locations in specific scenes.
    expected_location = ExpectedLocation(target_data, [
        target_data, large_container_data_1, large_container_data_2,
        large_container_data_3
    ], large_container_data_1)

    scene_dict = map_id_to_scene(hypercube.get_scenes())
    base_filename = save_scene_debug_files(scene_dict, 'container')

    for i in ['a', 'd', 'g', 'j', 'm', 'p']:
        for j in [i + '1', i + '2']:
            scene, index = scene_dict[j]
            print(f'[DEBUG] SCENE ID={j} INDEX={index}')
            assert verify_scene_properties(
                hypercube,
                scene,
                index,
                # Containers may hold target (will be tested later).
                ignore_parent=True,
                # Containers may not be in each scene (will be tested later).
                ignore_container_location=True
            )

            objects = scene.objects
            target = get_object(target_id, objects)
            large_container_1 = get_object(large_container_id_1, objects)
            large_container_2 = get_object(large_container_id_2, objects, True)
            large_container_3 = get_object(large_container_id_3, objects, True)

            # Verify 1st large container always at same location.
            target_location_name = target_data.location_plan_list[index].name
            assert verify_same_location(
                'container',
                expected_location.get(
                    large_container_id_1,
                    'random',
                    target_location_name,
                    j.endswith('2')
                ),
                get_position(large_container_data_1.instance_list[index]),
                get_position(large_container_1)
            )

            # Verify target is inside of 1st large container.
            if i in ['a', 'b', 'c', 'g', 'h', 'i', 'm', 'n', 'o']:
                assert verify_same_location(
                    'target',
                    expected_location.get(
                        target_id,
                        'inside_0',
                        is_receptacle_untrained=(j.endswith('2'))
                    ),
                    get_position(hypercube._target_data.instance_list[index]),
                    get_position(target)
                )
                assert verify_same_parent(
                    large_container_data_1.instance_list[index],
                    large_container_1,
                    hypercube._target_data.instance_list[index],
                    target
                )
                assert scene.goal['sceneInfo']['contained']['target']
                assert not scene.goal['sceneInfo']['uncontained']['target']

            # Verify target is close to 1st large container.
            if i in ['d', 'e', 'f', 'j', 'k', 'l', 'p', 'q', 'r']:
                assert verify_target_obstruct_location(
                    target_data.instance_list[index],
                    large_container_data_1.instance_list[index],
                    hypercube._performer_start
                )
                assert verify_same_location(
                    'target',
                    expected_location.get(target_id, 'close'),
                    get_position(hypercube._target_data.instance_list[index]),
                    get_position(target)
                )
                assert verify_not_parent(
                    large_container_data_1.instance_list[index],
                    large_container_1,
                    hypercube._target_data.instance_list[index],
                    target
                )
                assert not scene.goal['sceneInfo']['contained']['target']
                assert scene.goal['sceneInfo']['uncontained']['target']

            # Verify large container 2 location same across all scenes.
            if i in [
                'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l'
            ]:
                assert large_container_2
                assert verify_same_location(
                    'container',
                    expected_location.get(
                        large_container_id_2,
                        'random',
                        target_location_name,
                        j.endswith('2')
                    ),
                    get_position(large_container_data_2.instance_list[index]),
                    get_position(large_container_2)
                )
                assert verify_not_parent(
                    large_container_data_2.instance_list[index],
                    large_container_2
                )

            # Verify large container 2 not in scene.
            if i in ['m', 'n', 'o', 'p', 'q', 'r']:
                assert not large_container_2
                assert not large_container_data_2.instance_list[index]

            # Verify large container 3 location same across all scenes.
            if i in ['a', 'b', 'c', 'd', 'e', 'f']:
                assert large_container_3
                assert verify_same_location(
                    'container',
                    expected_location.get(
                        large_container_id_3,
                        'random',
                        target_location_name,
                        j.endswith('2')
                    ),
                    get_position(large_container_data_3.instance_list[index]),
                    get_position(large_container_3)
                )
                assert verify_not_parent(
                    large_container_data_3.instance_list[index],
                    large_container_3
                )

            # Verify large container 3 not in scene.
            if i in [
                'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r'
            ]:
                assert not large_container_3
                assert not large_container_data_3.instance_list[index]

            # Verify container trained shape.
            if j.endswith('1'):
                assert not large_container_data_1.untrained_plan_list[index]
                assert not large_container_data_2.untrained_plan_list[index]
                assert not large_container_data_3.untrained_plan_list[index]

            # Verify container untrained shape.
            if j.endswith('2'):
                assert large_container_data_1.untrained_plan_list[index]
                assert large_container_data_2.untrained_plan_list[index]
                assert large_container_data_3.untrained_plan_list[index]

    delete_scene_debug_files(scene_dict, base_filename)


@pytest.mark.slow
def test_obstacle_hypercube():
    hypercube_factory = InteractiveObstacleEvaluationHypercubeFactory(
        RetrievalGoal('obstacle')
    )
    hypercube = hypercube_factory._build(STARTER_SCENE)

    assert len(hypercube._data['obstacle']) == 1

    assert len(hypercube._data['large_container']) == 0
    assert len(hypercube._data['small_container']) == 0
    assert len(hypercube._data['occluder']) == 0

    assert 0 <= len(hypercube._small_context_object_list) <= 10

    target_data = hypercube._target_data
    target_id = target_data.trained_template['id']
    obstacle_data_1 = hypercube._data['obstacle'][0]
    obstacle_id_1 = obstacle_data_1.trained_template['id']

    # Save a location of each variable-location object, to verify that the
    # objects will always keep the same locations in specific scenes.
    expected_location = ExpectedLocation(target_data, [
        target_data, obstacle_data_1
    ])

    scene_dict = map_id_to_scene(hypercube.get_scenes())
    base_filename = save_scene_debug_files(scene_dict, 'obstacle')

    for i in ['a', 'b', 'c', 'd']:
        for j in [i + '1', i + '2']:
            scene, index = scene_dict[j]
            print(f'[DEBUG] SCENE ID={j} INDEX={index}')
            assert verify_scene_properties(
                hypercube,
                scene,
                index,
                # Obstacles should change location (will be tested later).
                ignore_obstacle_location=True
            )

            target = get_object(target_id, scene.objects)
            obstacle_1 = get_object(obstacle_id_1, scene.objects)

            # Verify target is in back of performer start.
            if i in ['a', 'b']:
                assert verify_not_immediately_visible(
                    hypercube._performer_start,
                    scene.objects,
                    target_data.instance_list[index],
                    obstacle_data_1.instance_list[index]
                )
                assert verify_same_location(
                    'target',
                    expected_location.get(target_id, 'back'),
                    get_position(target_data.instance_list[index]),
                    get_position(target)
                )

            # Verify target is in front of performer start.
            if i in ['c', 'd']:
                assert verify_immediately_visible(
                    hypercube._performer_start,
                    scene.objects,
                    target_data.instance_list[index],
                    obstacle_data_1.instance_list[index]
                )
                assert verify_same_location(
                    'target',
                    expected_location.get(target_id, 'front'),
                    get_position(target_data.instance_list[index]),
                    get_position(target)
                )

            # Verify obstacle is between target and performer start.
            if i in ['a', 'c']:
                assert verify_location_obstruct_target(
                    obstacle_data_1.instance_list[index],
                    target_data.instance_list[index],
                    hypercube._performer_start
                )
                assert verify_same_location(
                    'obstacle',
                    expected_location.get(
                        obstacle_id_1,
                        'between',
                        'front' if i == 'c' else 'back',
                        j.endswith('2')
                    ),
                    get_position(obstacle_data_1.instance_list[index]),
                    get_position(obstacle_1)
                )

                expected_distance = geometry.MAX_REACH_DISTANCE - min(
                    obstacle_1['debug']['dimensions']['x'],
                    obstacle_1['debug']['dimensions']['z']
                )
                target_poly = target['shows'][0]['boundingBox'].polygon_xz
                obstacle_poly = (
                    obstacle_1['shows'][0]['boundingBox'].polygon_xz
                )
                actual_distance = target_poly.distance(obstacle_poly)
                if actual_distance <= expected_distance:
                    print(
                        f'[ERROR] OBSTACLE SHOULD BE AT LEAST '
                        f'{expected_distance} AWAY FROM TARGET BUT IS '
                        f'{actual_distance}:\n{target=}\n{obstacle_1=}'
                    )
                    return False

            # Verify target is between obstacle and performer start.
            if i in ['b', 'd']:
                assert verify_target_obstruct_location(
                    target_data.instance_list[index],
                    obstacle_data_1.instance_list[index],
                    hypercube._performer_start
                )
                assert verify_same_location(
                    'obstacle',
                    expected_location.get(
                        obstacle_id_1,
                        'close',
                        'front' if i == 'd' else 'back',
                        j.endswith('2')
                    ),
                    get_position(obstacle_data_1.instance_list[index]),
                    get_position(obstacle_1)
                )

            # Verify obstacle trained shape.
            if j.endswith('1'):
                assert not obstacle_data_1.untrained_plan_list[index]

            # Verify obstacle untrained shape.
            if j.endswith('2'):
                assert obstacle_data_1.untrained_plan_list[index]

    delete_scene_debug_files(scene_dict, base_filename)


@pytest.mark.slow
def test_occluder_hypercube():
    hypercube_factory = InteractiveOccluderEvaluationHypercubeFactory(
        RetrievalGoal('occluder')
    )
    hypercube = hypercube_factory._build(STARTER_SCENE)

    assert len(hypercube._data['occluder']) == 3

    assert len(hypercube._data['large_container']) == 0
    assert len(hypercube._data['small_container']) == 0
    assert len(hypercube._data['obstacle']) == 0

    assert 0 <= len(hypercube._small_context_object_list) <= 10

    target_data = hypercube._target_data
    target_id = target_data.trained_template['id']
    occluder_data_1 = hypercube._data['occluder'][0]
    occluder_data_2 = hypercube._data['occluder'][1]
    occluder_data_3 = hypercube._data['occluder'][2]
    occluder_id_1 = occluder_data_1.trained_template['id']
    occluder_id_2 = occluder_data_2.trained_template['id']
    occluder_id_3 = occluder_data_3.trained_template['id']

    # Save a location of each variable-location object, to verify that the
    # objects will always keep the same locations in specific scenes.
    expected_location = ExpectedLocation(target_data, [
        target_data, occluder_data_1, occluder_data_2, occluder_data_3
    ])

    scene_dict = map_id_to_scene(hypercube.get_scenes())
    base_filename = save_scene_debug_files(scene_dict, 'occluder')

    for i in ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l']:
        for j in [i + '1', i + '2']:
            scene, index = scene_dict[j]
            print(f'[DEBUG] SCENE ID={j} INDEX={index}')
            assert verify_scene_properties(
                hypercube,
                scene,
                index,
                # Occluders should change location (will be tested later).
                ignore_occluder_location=True
            )

            target = get_object(target_id, scene.objects)
            occluder_1 = get_object(occluder_id_1, scene.objects)
            occluder_2 = get_object(occluder_id_2, scene.objects, True)
            occluder_3 = get_object(occluder_id_3, scene.objects, True)

            # Verify target is in back of performer start.
            if i in ['a', 'b', 'e', 'f', 'i', 'j']:
                assert verify_not_immediately_visible(
                    hypercube._performer_start,
                    scene.objects,
                    target_data.instance_list[index],
                    occluder_data_1.instance_list[index]
                )
                assert verify_same_location(
                    'target',
                    expected_location.get(target_id, 'back'),
                    get_position(target_data.instance_list[index]),
                    get_position(target)
                )

            # Verify target is in front of performer start.
            if i in ['c', 'd', 'g', 'h', 'k', 'l']:
                assert verify_immediately_visible(
                    hypercube._performer_start,
                    scene.objects,
                    target_data.instance_list[index],
                    occluder_data_1.instance_list[index]
                )
                assert verify_same_location(
                    'target',
                    expected_location.get(target_id, 'front'),
                    get_position(target_data.instance_list[index]),
                    get_position(target)
                )

            # Verify occluder 1 is between target and performer start.
            if i in ['a', 'c', 'e', 'g', 'i', 'k']:
                assert verify_location_obstruct_target(
                    occluder_data_1.instance_list[index],
                    target_data.instance_list[index],
                    hypercube._performer_start
                )
                assert verify_same_location(
                    'occluder',
                    expected_location.get(
                        occluder_id_1,
                        'between',
                        'front' if i in ['c', 'g', 'k'] else 'back',
                        j.endswith('2')
                    ),
                    get_position(occluder_data_1.instance_list[index]),
                    get_position(occluder_1)
                )

                expected_distance = geometry.MAX_REACH_DISTANCE
                target_poly = target['shows'][0]['boundingBox'].polygon_xz
                occluder_poly = (
                    occluder_1['shows'][0]['boundingBox'].polygon_xz
                )
                actual_distance = target_poly.distance(occluder_poly)
                if actual_distance > expected_distance:
                    print(
                        f'[ERROR] OCCLUDER SHOULD BE AT MOST '
                        f'{expected_distance} AWAY FROM TARGET BUT IS '
                        f'{actual_distance}:\n{target=}\n{occluder_1=}'
                    )
                    return False

            # Verify target is between occluder 1 and performer start.
            if i in ['b', 'd', 'f', 'h', 'j', 'l']:
                assert verify_target_obstruct_location(
                    target_data.instance_list[index],
                    occluder_data_1.instance_list[index],
                    hypercube._performer_start
                )
                assert verify_same_location(
                    'occluder',
                    expected_location.get(
                        occluder_id_1,
                        'close',
                        'front' if i in ['d', 'h', 'l'] else 'back',
                        j.endswith('2')
                    ),
                    get_position(occluder_data_1.instance_list[index]),
                    get_position(occluder_1)
                )

            # Verify occluder 2 location same across all scenes.
            if i in ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h']:
                assert occluder_2
                assert verify_same_location(
                    'occluder',
                    expected_location.get(
                        occluder_id_2,
                        'random',
                        'front' if i in ['c', 'd', 'g', 'h'] else 'back',
                        j.endswith('2')
                    ),
                    get_position(occluder_data_2.instance_list[index]),
                    get_position(occluder_2)
                )

            # Verify occluder 2 not in scene.
            if i in ['i', 'j', 'k', 'l']:
                assert not occluder_2
                assert not occluder_data_2.instance_list[index]

            # Verify occluder 3 location same across all scenes.
            if i in ['a', 'b', 'c', 'd']:
                assert occluder_3
                assert verify_same_location(
                    'occluder',
                    expected_location.get(
                        occluder_id_3,
                        'random',
                        'front' if i in ['c', 'd'] else 'back',
                        j.endswith('2')
                    ),
                    get_position(occluder_data_3.instance_list[index]),
                    get_position(occluder_3)
                )

            # Verify occluder 3 not in scene.
            if i in ['e', 'f', 'g', 'h', 'i', 'j', 'k', 'l']:
                assert not occluder_3
                assert not occluder_data_3.instance_list[index]

            # Verify occluder trained shape.
            if j.endswith('1'):
                assert not occluder_data_1.untrained_plan_list[index]
                assert not occluder_data_2.untrained_plan_list[index]
                assert not occluder_data_3.untrained_plan_list[index]

            # Verify occluder untrained shape.
            if j.endswith('2'):
                assert occluder_data_1.untrained_plan_list[index]
                assert occluder_data_2.untrained_plan_list[index]
                assert occluder_data_3.untrained_plan_list[index]

    delete_scene_debug_files(scene_dict, base_filename)
