Installation
============

.. tab-set::

    .. tab-item:: Stable (with conda)

        This is our recommend installation method! Follow the steps below to start using ``elk``!

        #. :download:`Download the environment.yml file from our repository <https://raw.githubusercontent.com/tobin-wainer/elk/main/environment.yml>`
        #. Create a new conda environment using this file::

                conda env create -f path/to/environment.yml

        #. Activate the environment by running::

                conda activate elk
        
        #. Pip install elk!::
                
                pip install astro-elk

        and you should be all set! Now it's time to learn how about `Getting Started <https://elk.readthedocs.io/en/latest/pages/getting_started.html>`__ with ``elk``.

        .. note::
            Please note, `elk` is supported for python 3.9+ environments. 

        .. note::
            If you also want to work with Jupyter notebooks then you'll also need to install jupyter/ipython to this environment!

    .. tab-item:: Stable (without conda)

        We don't recommend installing ``elk`` without a conda environment but if you prefer to do it this
        way then all you need to do is run::

            pip install astro-elk

        and you should be all set! Now it's time to learn how about `Getting Started <getting_started>`__ with ``elk``.

        .. note::
            If you also want to work with Jupyter notebooks then you'll also need to install jupyter/ipython to this environment!
    .. tab-item:: Development (from GitHub)
        
        .. warning::

            We don't guarantee that there won't be mistakes or bugs in the development version, use at your own risk!

        The latest development version is available directly from our `GitHub Repo
        <https://github.com/tobin-wainer/elk>`_. To start, clone the repository onto your machine: ::
        
            git clone https://github.com/tobin-wainer/elk
            cd elk

        Next, we recommend that you create a Conda environment for working with ``elk``.
        You can do this by running::

            conda env create -f environment.yml

        And then activate the environment by running::

            conda activate elk

        At this point, all that's left to do is install ``elk``!::

            pip install .

        and you should be all set! Now it's time to learn how about `Getting Started <https://elk.readthedocs.io/en/latest/pages/getting_started.html>`__ with ``elk``.

        .. note::
            If you also want to work with Jupyter notebooks then you'll also need to install jupyter/ipython to this environment!
