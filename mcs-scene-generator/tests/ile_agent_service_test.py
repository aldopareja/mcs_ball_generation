import math

import pytest
from machine_common_sense.config_manager import Vector3d

from generator.agents import AGENT_MOVEMENT_ANIMATIONS, AGENT_TYPES
from generator.scene import Scene
from ideal_learning_env.agent_service import (
    AgentActionConfig,
    AgentConfig,
    AgentCreationService,
    AgentMovementConfig,
    AgentSettings,
    get_default_agent_settings
)
from ideal_learning_env.defs import ILEConfigurationException, ILEException
from ideal_learning_env.interactable_object_service import (
    KeywordLocationConfig
)
from ideal_learning_env.numerics import MinMaxFloat, VectorFloatConfig
from ideal_learning_env.object_services import (
    KeywordLocation,
    ObjectRepository
)
from ideal_learning_env.structural_object_service import (
    StructuralPlatformConfig,
    StructuralPlatformCreationService
)
from tests.ile_helper import (
    prior_scene,
    prior_scene_custom_size,
    prior_scene_with_target
)


@pytest.fixture(autouse=True)
def run_around_test():
    # Prepare test
    ObjectRepository.get_instance().clear()

    # Run test
    yield

    # Cleanup
    ObjectRepository.get_instance().clear()


def test_agent_service_reconcile():
    scene = Scene()
    template = AgentConfig(
        1,
        type=[
            'test_type',
            'test_type2'],
        agent_settings=AgentSettings(
            chest=[
                2,
                4]),
        position=VectorFloatConfig([1, 2], MinMaxFloat(0.5, 0.6),
                                   [1, MinMaxFloat(4.4, 4.5)]),
        rotation_y=[56, 57])
    srv = AgentCreationService()
    reconciled: AgentConfig = srv.reconcile(scene, template)
    assert reconciled.num == 1
    assert reconciled.type in ['test_type', 'test_type2']
    assert reconciled.agent_settings.chest in [2, 4]
    assert reconciled.position.x in [1, 2]
    assert 0.5 <= reconciled.position.y <= 0.6
    assert reconciled.position.z == 1 or 4.4 <= reconciled.position.z <= 4.5
    assert reconciled.rotation_y in [56, 57]
    assert reconciled.movement is None
    assert reconciled.actions == []


def test_agent_service_reconcile_default():
    scene = Scene()
    template = AgentConfig(1)
    srv = AgentCreationService()
    reconciled: AgentConfig = srv.reconcile(scene, template)
    assert reconciled.num == 1
    assert reconciled.type in AGENT_TYPES
    assert -5 <= reconciled.position.x <= 5
    assert 0 == reconciled.position.y
    assert -5 <= reconciled.position.z <= 5
    assert 0 <= reconciled.rotation_y <= 360
    assert reconciled.movement is None
    assert reconciled.actions == []


def test_agent_service_create():
    scene = Scene()
    template = AgentConfig(
        1,
        type='test_type',
        agent_settings=AgentSettings(),
        position=VectorFloatConfig(1, 0.5, 1),
        rotation_y=90)
    srv = AgentCreationService()
    agent = srv.create_feature_from_specific_values(
        scene, template, template)
    assert agent['type'] == 'test_type'
    assert agent['id'].startswith('agent')
    assert agent['agentSettings']
    assert agent['shows'][0]['position']['x'] == 1
    assert agent['shows'][0]['position']['y'] == 0.5
    assert agent['shows'][0]['position']['z'] == 1
    assert agent['shows'][0]['rotation']['x'] == 0
    assert agent['shows'][0]['rotation']['y'] == 90
    assert agent['shows'][0]['rotation']['z'] == 0
    assert agent['shows'][0]['scale']['x'] == 1
    assert agent['shows'][0]['scale']['y'] == 1
    assert agent['shows'][0]['scale']['z'] == 1
    bb = agent['shows'][0]['boundingBox']
    assert bb.min_y == 0.5
    assert bb.max_y == 1.46
    assert bb.box_xz[0] == Vector3d(x=1.15, y=0, z=0.85)
    assert bb.box_xz[1] == Vector3d(x=0.85, y=0, z=0.85)
    assert bb.box_xz[2] == Vector3d(x=0.85, y=0, z=1.15)
    assert bb.box_xz[3] == Vector3d(x=1.15, y=0, z=1.15)

    assert agent['debug']['dimensions'] == {'x': 0.3, 'y': 0.96, 'z': 0.3}


def test_agent_service_add():
    scene = Scene()
    template = AgentConfig(
        1,
        type='test_type',
        agent_settings=AgentSettings(),
        position=VectorFloatConfig(1, 0.5, 1),
        rotation_y=90)
    srv = AgentCreationService()
    agents, _ = srv.add_to_scene(scene, template, [])
    agent = agents[0]
    assert agent['type'] == 'test_type'
    assert agent['id'].startswith('agent')
    assert agent['agentSettings']
    assert agent['shows'][0]['position']['x'] == 1
    assert agent['shows'][0]['position']['y'] == 0.5
    assert agent['shows'][0]['position']['z'] == 1
    assert agent['shows'][0]['rotation']['x'] == 0
    assert agent['shows'][0]['rotation']['y'] == 90
    assert agent['shows'][0]['rotation']['z'] == 0
    assert agent['shows'][0]['scale']['x'] == 1
    assert agent['shows'][0]['scale']['y'] == 1
    assert agent['shows'][0]['scale']['z'] == 1
    bb = agent['shows'][0]['boundingBox']
    assert bb.min_y == 0.5
    assert bb.max_y == 1.46
    assert bb.box_xz[0] == Vector3d(x=1.15, y=0, z=0.85)
    assert bb.box_xz[1] == Vector3d(x=0.85, y=0, z=0.85)
    assert bb.box_xz[2] == Vector3d(x=0.85, y=0, z=1.15)
    assert bb.box_xz[3] == Vector3d(x=1.15, y=0, z=1.15)

    assert agent['debug']['dimensions'] == {'x': 0.3, 'y': 0.96, 'z': 0.3}
    repo_agents = (ObjectRepository.get_instance().
                   get_all_from_labeled_objects('agent'))
    assert len(repo_agents) == 1
    assert repo_agents[0].instance == agents[0]


def test_agent_service_add_fail():
    scene = Scene()
    template = AgentConfig(
        2,
        type='test_type',
        agent_settings=AgentSettings(),
        position=VectorFloatConfig(1, 0.5, 1),
        rotation_y=90)
    srv = AgentCreationService()
    bounds = []
    srv.add_to_scene(scene, template, bounds)
    with pytest.raises(ILEException):
        srv.add_to_scene(scene, template, bounds)


def test_agent_service_add_actions():
    scene = Scene()
    actions = [
        AgentActionConfig(id='action2', step_begin=10),
        AgentActionConfig(id='action1', step_begin=2, step_end=7),
        AgentActionConfig(id='action3', step_begin=20, is_loop_animation=True)
    ]
    template = AgentConfig(
        1,
        type='test_type',
        agent_settings=AgentSettings(),
        position=VectorFloatConfig(0, 0, 2),
        rotation_y=180,
        actions=actions)
    srv = AgentCreationService()
    agents, _ = srv.add_to_scene(scene, template, [])
    agent = agents[0]
    assert agent['type'] == 'test_type'
    assert agent['id'].startswith('agent')
    assert agent['agentSettings']
    assert agent['shows'][0]['position']['x'] == 0
    assert agent['shows'][0]['position']['y'] == 0
    assert agent['shows'][0]['position']['z'] == 2
    assert agent['shows'][0]['rotation']['x'] == 0
    assert agent['shows'][0]['rotation']['y'] == 180
    assert agent['shows'][0]['rotation']['z'] == 0
    assert agent['shows'][0]['scale']['x'] == 1
    assert agent['shows'][0]['scale']['y'] == 1
    assert agent['shows'][0]['scale']['z'] == 1
    bb = agent['shows'][0]['boundingBox']
    assert bb.min_y == 0
    assert bb.max_y == 0.96
    assert bb.box_xz[0].x == pytest.approx(-0.15)
    assert bb.box_xz[0].y == 0
    assert bb.box_xz[0].z == 1.85
    assert bb.box_xz[1].x == pytest.approx(-0.15)
    assert bb.box_xz[1].y == 0
    assert bb.box_xz[1].z == 2.15
    assert bb.box_xz[2].x == pytest.approx(0.15)
    assert bb.box_xz[2].y == 0
    assert bb.box_xz[2].z == 2.15
    assert bb.box_xz[3].x == pytest.approx(0.15)
    assert bb.box_xz[3].y == 0
    assert bb.box_xz[3].z == 1.85

    assert len(agent['actions']) == 3
    assert agent['actions'][0]['id'] == 'action1'
    assert agent['actions'][0]['stepBegin'] == 2
    assert agent['actions'][0].get('stepEnd') == 7
    assert agent['actions'][0].get('isLoopAnimation') is False
    assert agent['actions'][1]['id'] == 'action2'
    assert agent['actions'][1]['stepBegin'] == 10
    assert agent['actions'][1].get('stepEnd') is None
    assert agent['actions'][1].get('isLoopAnimation') is False
    assert agent['actions'][2]['id'] == 'action3'
    assert agent['actions'][2]['stepBegin'] == 20
    assert agent['actions'][2].get('stepEnd') is None
    assert agent['actions'][2].get('isLoopAnimation') is True

    assert agent['debug']['dimensions'] == {'x': 0.3, 'y': 0.96, 'z': 0.3}
    repo_agents = (ObjectRepository.get_instance().
                   get_all_from_labeled_objects('agent'))
    assert len(repo_agents) == 1
    assert repo_agents[0].instance == agents[0]


def test_agent_service_add_actions_fail():
    scene = Scene()
    actions = [
        AgentActionConfig(id='action2', step_begin=1),
        AgentActionConfig(id='action1', step_begin=1, step_end=7),
        AgentActionConfig(id='action3', step_begin=20, is_loop_animation=True)
    ]
    template = AgentConfig(
        1,
        type='test_type',
        agent_settings=AgentSettings(),
        position=VectorFloatConfig(0, 0, 2),
        rotation_y=180,
        actions=actions)
    srv = AgentCreationService()
    with pytest.raises(ILEException):
        srv.add_to_scene(scene, template, [])


def test_agent_service_reconcile_movement_all():
    scene = Scene()
    template = AgentConfig(1)
    template.movement = AgentMovementConfig(
        animation=['anim1', 'anim2'], step_begin=[3, 4],
        points=[VectorFloatConfig(1, 0, 3),
                VectorFloatConfig([2, 3], 0, [4, 5]),
                VectorFloatConfig(MinMaxFloat(1.1, 1.2), 0,
                                  MinMaxFloat(3.3, 3.4))],
        bounds=[VectorFloatConfig(3, 0, 1),
                VectorFloatConfig([-1, 1], 0, [-2, -1]),
                VectorFloatConfig(MinMaxFloat(-1.2, -1.1), 0,
                                  MinMaxFloat(3.3, 3.4))],
        num_points=[3, 4],
        repeat=[True, True])
    srv = AgentCreationService()
    reconciled: AgentConfig = srv.reconcile(scene, template)
    assert reconciled.num == 1
    move = reconciled.movement
    assert move.animation in ['anim1', 'anim2']
    assert move.step_begin in [3, 4]
    ps = move.points
    assert ps[0].x == 1
    assert ps[0].y == 0
    assert ps[0].z == 3
    assert ps[1].x in [2, 3]
    assert ps[1].y == 0
    assert ps[1].z in [4, 5]
    assert 1.1 <= ps[2].x <= 1.2
    assert ps[2].y == 0
    assert 3.3 <= ps[2].z <= 3.4
    bs = move.bounds
    assert bs[0].x == 3
    assert bs[0].y == 0
    assert bs[0].z == 1
    assert bs[1].x in [-1, 1]
    assert bs[1].y == 0
    assert bs[1].z in [-2, -1]
    assert -1.2 <= bs[2].x <= -1.1
    assert bs[2].y == 0
    assert 3.3 <= bs[2].z <= 3.4
    assert move.num_points in [3, 4]
    assert move.repeat is True


def test_agent_service_reconcile_movement_empty():
    scene = Scene()
    template = AgentConfig(1)
    template.movement = AgentMovementConfig()
    srv = AgentCreationService()
    reconciled: AgentConfig = srv.reconcile(scene, template)
    assert reconciled.num == 1
    move = reconciled.movement
    assert move.animation in AGENT_MOVEMENT_ANIMATIONS
    assert 0 <= move.step_begin <= 20
    assert move.points is None
    assert move.bounds is None
    assert 0 <= move.num_points <= 20
    assert move.repeat in [True, False]


def test_agent_service_create_movement_bounds_and_points():
    scene = Scene()
    template = AgentConfig(
        1, position=Vector3d(), rotation_y=0,
        agent_settings=get_default_agent_settings())
    template.movement = AgentMovementConfig(
        animation='anim1', step_begin=3,
        points=[Vector3d(x=1, y=0, z=3),
                Vector3d(x=2, y=0, z=4)],
        bounds=[Vector3d(x=-3, y=0, z=-1),
                Vector3d(x=-1, y=0, z=-2),
                Vector3d(x=-1.2, y=0, z=-3.3)],
        num_points=3,
        repeat=True)

    srv = AgentCreationService()
    agent = srv.create_feature_from_specific_values(
        scene, template, template)
    move = agent['agentMovement']
    assert agent
    assert move
    assert move['repeat'] is True
    assert move['stepBegin'] == 3
    seq = move['sequence']
    assert seq[0]['animation'] == 'anim1'
    assert seq[0]['endPoint'] == {'x': 1.0, 'z': 3.0}
    assert seq[1]['animation'] == 'anim1'
    assert seq[1]['endPoint'] == {'x': 2.0, 'z': 4.0}


def test_agent_service_create_movement_bounds():
    scene = Scene()
    template = AgentConfig(
        1, position=Vector3d(), rotation_y=0,
        agent_settings=get_default_agent_settings())
    template.movement = AgentMovementConfig(
        animation='anim2', step_begin=1,
        bounds=[Vector3d(x=0, y=0, z=1.0),
                Vector3d(x=.2, y=0, z=1.0),
                Vector3d(x=.2, y=0, z=1.3),
                Vector3d(x=0, y=0, z=1.3)],
        num_points=10,
        repeat=True)

    srv = AgentCreationService()
    agent = srv.create_feature_from_specific_values(
        scene, template, template)
    move = agent['agentMovement']
    assert agent
    assert move
    assert move['repeat'] is True
    assert move['stepBegin'] == 1
    seq = move['sequence']
    assert len(seq) == 10
    for item in seq:
        assert item['animation'] == 'anim2'
        assert 0 <= item['endPoint']['x'] <= 0.2
        assert 1.0 <= item['endPoint']['z'] <= 1.3


def test_agent_service_create_no_bounds():
    scene = prior_scene_custom_size(2, 10)
    template = AgentConfig(
        1, position=Vector3d(), rotation_y=0,
        agent_settings=get_default_agent_settings())
    template.movement = AgentMovementConfig(
        animation='anim3', step_begin=3,
        num_points=10,
        repeat=False)

    srv = AgentCreationService()
    agent = srv.create_feature_from_specific_values(
        scene, template, template)
    move = agent['agentMovement']
    assert agent
    assert move
    assert move['repeat'] is False
    assert move['stepBegin'] == 3
    seq = move['sequence']
    assert len(seq) == 10
    for item in seq:
        assert item['animation'] == 'anim3'
        assert -1 <= item['endPoint']['x'] <= 1
        assert -5 <= item['endPoint']['z'] <= 5


def test_agent_service_create_no_data():
    scene = prior_scene_custom_size(2, 10)
    template = AgentConfig(
        1, position=Vector3d(), rotation_y=0,
        agent_settings=get_default_agent_settings())
    template.movement = AgentMovementConfig()

    srv = AgentCreationService()
    agent = srv.create_feature_from_specific_values(
        scene, template, template)
    move = agent['agentMovement']
    assert agent
    assert move
    assert move['repeat'] in [True, False]
    assert move['stepBegin'] > -1
    seq = move['sequence']
    for item in seq:
        assert item['animation'] in AGENT_MOVEMENT_ANIMATIONS
        assert -1 <= item['endPoint']['x'] <= 1
        assert -5 <= item['endPoint']['z'] <= 5


def test_agent_opposite_x():
    scene = prior_scene_with_target(add_to_repo=True)
    template = AgentConfig(
        1, keyword_location=KeywordLocationConfig(
            keyword=KeywordLocation.OPPOSITE_X,
            relative_object_label='target'),
        rotation_y=0,
        agent_settings=get_default_agent_settings())

    srv = AgentCreationService()
    agents, _ = srv.add_to_scene(scene, template, [])
    agent = agents[0]
    assert agent
    assert agent['shows'][0]['position']['x'] == 1.03
    assert agent['shows'][0]['position']['y'] == 0
    assert agent['shows'][0]['position']['z'] == 4.08


def test_agent_opposite_z():
    scene = prior_scene_with_target(add_to_repo=True)
    template = AgentConfig(
        1, keyword_location=KeywordLocationConfig(
            keyword=KeywordLocation.OPPOSITE_Z,
            relative_object_label='target'),
        rotation_y=0,
        agent_settings=get_default_agent_settings())

    srv = AgentCreationService()
    agents, _ = srv.add_to_scene(scene, template, [])
    agent = agents[0]
    assert agent
    assert agent['shows'][0]['position']['x'] == -1.03
    assert agent['shows'][0]['position']['y'] == 0
    assert agent['shows'][0]['position']['z'] == -4.08


def test_agent_adjacent_to_object():
    scene = prior_scene_with_target(add_to_repo=True)
    template = AgentConfig(
        1, keyword_location=KeywordLocationConfig(
            keyword=KeywordLocation.ADJACENT_TO_OBJECT,
            relative_object_label='target'),
        rotation_y=0,
        agent_settings=get_default_agent_settings())

    srv = AgentCreationService()
    agents, _ = srv.add_to_scene(scene, template, [])
    agent = agents[0]
    assert agent
    x = agent['shows'][0]['position']['x']
    z = agent['shows'][0]['position']['z']
    dist = math.dist((x, z), (-1.03, 4.08))
    assert dist < 0.9


def test_agent_behind_object_from_performer():
    scene = prior_scene_with_target(add_to_repo=True)
    template = AgentConfig(
        1, keyword_location=KeywordLocationConfig(
            keyword=KeywordLocation.BEHIND_OBJECT_FROM_PERFORMER,
            relative_object_label='target'),
        rotation_y=0,
        agent_settings=get_default_agent_settings())

    srv = AgentCreationService()
    agents, _ = srv.add_to_scene(scene, template, [])
    agent = agents[0]
    assert agent
    assert -2 < agent['shows'][0]['position']['x'] < 0
    assert agent['shows'][0]['position']['y'] == 0
    assert agent['shows'][0]['position']['z'] > 4.08


def test_agent_front_of_performer():
    scene = prior_scene_with_target(add_to_repo=True)
    template = AgentConfig(
        1, keyword_location=KeywordLocationConfig(
            keyword=KeywordLocation.FRONT_OF_PERFORMER,
            relative_object_label='target'),
        rotation_y=0,
        agent_settings=get_default_agent_settings())

    srv = AgentCreationService()
    agents, _ = srv.add_to_scene(scene, template, [])
    agent = agents[0]
    assert agent
    assert agent['shows'][0]['position']['x'] == 0
    assert agent['shows'][0]['position']['y'] == 0
    assert agent['shows'][0]['position']['z'] > 0


def test_agent_on_object():
    scene = prior_scene()
    bounds = []
    plat_template = StructuralPlatformConfig(
        num=1, position=VectorFloatConfig(1, 0, 2), scale=1)
    StructuralPlatformCreationService().add_to_scene(
        scene, plat_template, bounds)
    template = AgentConfig(
        1, keyword_location=KeywordLocationConfig(
            keyword=KeywordLocation.ON_OBJECT,
            relative_object_label='platforms'),
        rotation_y=0,
        agent_settings=get_default_agent_settings())

    srv = AgentCreationService()
    agents, _ = srv.add_to_scene(scene, template, bounds)
    agent = agents[0]
    assert agent
    assert 0.5 < agent['shows'][0]['position']['x'] < 1.5
    assert agent['shows'][0]['position']['y'] == 1
    assert 1.5 < agent['shows'][0]['position']['z'] < 2.5


def test_agent_on_object_centered():
    scene = prior_scene()
    bounds = []
    plat_template = StructuralPlatformConfig(
        num=1, position=VectorFloatConfig(-3, 0, 1.5), scale=1)
    StructuralPlatformCreationService().add_to_scene(
        scene, plat_template, bounds)
    template = AgentConfig(
        1, keyword_location=KeywordLocationConfig(
            keyword=KeywordLocation.ON_OBJECT,
            relative_object_label='platforms'),
        rotation_y=0,
        agent_settings=get_default_agent_settings())

    srv = AgentCreationService()
    agents, _ = srv.add_to_scene(scene, template, [])
    agent = agents[0]
    assert agent
    assert -3.5 < agent['shows'][0]['position']['x'] < -2.5
    assert agent['shows'][0]['position']['y'] == 1
    assert 1 < agent['shows'][0]['position']['z'] < 2


def test_agent_keyword_location_random():
    scene = prior_scene_with_target(add_to_repo=True)
    template = AgentConfig(
        1, keyword_location=KeywordLocationConfig(
            keyword=KeywordLocation.RANDOM,
            relative_object_label='target'),
        rotation_y=0,
        agent_settings=get_default_agent_settings())

    srv = AgentCreationService()
    agents, _ = srv.add_to_scene(scene, template, [])
    agent = agents[0]
    assert agent
    assert -5 < agent['shows'][0]['position']['x'] < 5
    assert agent['shows'][0]['position']['y'] == 0
    assert -5 < agent['shows'][0]['position']['z'] < 5


def test_agent_keyword_location_and_position_fail():
    scene = prior_scene_with_target(add_to_repo=True)
    template = AgentConfig(
        1, keyword_location=KeywordLocationConfig(
            keyword=KeywordLocation.RANDOM,
            relative_object_label='target'),
        position=VectorFloatConfig(1, 2, 3),
        rotation_y=0,
        agent_settings=get_default_agent_settings())

    srv = AgentCreationService()
    with pytest.raises(ILEException):
        agents, _ = srv.add_to_scene(scene, template, [])


def test_agent_opposite_z_fail_overlap():
    scene = prior_scene_with_target(add_to_repo=True)
    template = AgentConfig(
        1, keyword_location=KeywordLocationConfig(
            keyword=KeywordLocation.OPPOSITE_Z,
            relative_object_label='target'),
        rotation_y=0,
        agent_settings=get_default_agent_settings())

    srv = AgentCreationService()
    bounds = []
    srv.add_to_scene(scene, template, bounds)
    with pytest.raises(ILEException):
        srv.add_to_scene(scene, template, bounds)


def test_agent_associate_with_agent_fail():
    scene = prior_scene_with_target(add_to_repo=True)
    template = AgentConfig(
        2, keyword_location=KeywordLocationConfig(
            keyword=KeywordLocation.ASSOCIATED_WITH_AGENT,
            relative_object_label='target'),
        rotation_y=0,
        agent_settings=get_default_agent_settings())

    srv = AgentCreationService()
    with pytest.raises(ILEConfigurationException):
        srv.add_to_scene(scene, template, [])


def test_agent_in_fail():
    scene = prior_scene_with_target(add_to_repo=True)
    template = AgentConfig(
        2, keyword_location=KeywordLocationConfig(
            keyword=KeywordLocation.IN_CONTAINER,
            relative_object_label='target',
            container_label="test"),
        rotation_y=0,
        agent_settings=get_default_agent_settings())

    srv = AgentCreationService()
    with pytest.raises(ILEConfigurationException):
        srv.add_to_scene(scene, template, [])


def test_agent_in_with_fail():
    scene = prior_scene_with_target(add_to_repo=True)
    template = AgentConfig(
        2, keyword_location=KeywordLocationConfig(
            keyword=KeywordLocation.IN_CONTAINER_WITH_OBJECT,
            relative_object_label='target',
            container_label="test"),
        rotation_y=0,
        agent_settings=get_default_agent_settings())

    srv = AgentCreationService()
    with pytest.raises(ILEConfigurationException):
        srv.add_to_scene(scene, template, [])


def test_agent_occlude_fail():
    scene = prior_scene_with_target(add_to_repo=True)
    template = AgentConfig(
        2, keyword_location=KeywordLocationConfig(
            keyword=KeywordLocation.OCCLUDE_OBJECT,
            relative_object_label='target'),
        rotation_y=0,
        agent_settings=get_default_agent_settings())

    srv = AgentCreationService()
    with pytest.raises(ILEConfigurationException):
        srv.add_to_scene(scene, template, [])
