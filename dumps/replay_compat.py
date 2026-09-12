# coding=utf-8
"""Patched replay script that handles numpy._core → numpy.core remapping.
   For dumps pickled with numpy >= 2.0 loaded on environments with numpy < 2.0.
   Usage: python dumps/replay_compat.py --trace_file dumps/episode_done_xxx.dump
"""

import sys
import io
import six.moves.cPickle as cPickle
import numpy as np

from gfootball.env import script_helpers
from absl import app
from absl import flags

FLAGS = flags.FLAGS
flags.DEFINE_string('trace_file', None, 'Trace file to replay')
flags.DEFINE_integer('fps', 10, 'How many frames per second to render')
flags.mark_flag_as_required('trace_file')


class NumpyCompatUnpickler(cPickle.Unpickler):
    """Unpickler that remaps numpy._core → numpy.core for cross-version compat."""

    _MODULE_MAP = {
        'numpy._core': 'numpy.core',
        'numpy._core.multiarray': 'numpy.core.multiarray',
        'numpy._core.umath': 'numpy.core.umath',
    }

    def find_class(self, module, name):
        mapped = self._MODULE_MAP.get(module, module)
        return super(NumpyCompatUnpickler, self).find_class(mapped, name)


def load_dump_compat(dump_file):
    """Load dump with numpy._core compat. Mirrors script_helpers.load_dump."""
    dump = []
    with open(dump_file, 'rb') as f:
        while True:
            try:
                step = NumpyCompatUnpickler(f).load()
            except EOFError:
                break
            except Exception:
                # Fallback: try raw unpickler
                break
            dump.append(step)
    return dump


def main(_):
    # Monkey-patch script_helpers to use our compat loader
    original_replay = script_helpers.ScriptHelpers.replay

    def patched_replay(self, dump, fps=10):
        replay_data = load_dump_compat(dump)
        if not replay_data:
            print("No steps loaded from dump.")
            return

        # Patch load_dump to return our compat-loaded data
        original_load = script_helpers.ScriptHelpers.load_dump
        script_helpers.ScriptHelpers.load_dump = lambda self, d: replay_data
        try:
            original_replay(self, dump, fps)
        finally:
            script_helpers.ScriptHelpers.load_dump = original_load

    script_helpers.ScriptHelpers.replay = patched_replay

    # Now run normal replay
    script_helpers.ScriptHelpers().replay(FLAGS.trace_file, FLAGS.fps)


if __name__ == '__main__':
    app.run(main)
