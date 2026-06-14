from glob import glob
import os

from setuptools import find_packages, setup

package_name = 'ros_pendulum'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob(os.path.join('launch', '*launch.[pxy][yma]*'))),
        (os.path.join('share', package_name, 'urdf'), glob(os.path.join('urdf', '*.urdf')) + glob(os.path.join('urdf', '*.xacro'))),
        (os.path.join('share', package_name, 'urdf', 'assets'), glob(os.path.join('urdf', 'assets', '*'))),
        (os.path.join('share', package_name, 'config'), glob(os.path.join('config', '*'))),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='arusso',
    maintainer_email='andrerusso02@gmail.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'encoder_hw_node = ros_pendulum.encoder_hw_node:main'
        ],
    },
)
