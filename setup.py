#!/usr/bin/env python3


from setuptools import setup

if __name__ == '__main__':
    setup(
        name='peft_ex',
        packages=[
            'peft_ex',
            'peft_ex.optim',
        ],
        version='0.2',
        description='PEFT Extensions.',
        long_description_content_type='text/markdown',
        long_description='PEFT Extensions',
        license='Apache License 2.0',
        author='xi',
        author_email='gylv@mail.ustc.edu.cn',
        platforms='any',
        classifiers=[
            'Programming Language :: Python :: 3.6',
            'Programming Language :: Python :: 3.7',
            'Programming Language :: Python :: 3.8',
        ],
        include_package_data=True,
        zip_safe=True,
        install_requires=[
            'peft'
        ]
    )
