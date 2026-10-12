from setuptools import find_packages, setup


package_name = 'primus_actions'

setup(
    name=package_name,
    version='0.0.1',
    packages=find_packages(exclude=['test']),
    data_files=[
        (
            'share/ament_index/resource_index/packages',
            ['resource/' + package_name],
        ),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    entry_points={
        'console_scripts': [
            'task_executor = primus_actions.task_executor:main',
            'perception_bridge = primus_actions.perception_bridge:main',
            'brain_bridge = primus_actions.brain_bridge:main',
        ],
    },
)
