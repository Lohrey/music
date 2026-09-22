#!/bin/bash
# Called by RunPod's /start.sh (runpod/pytorch 1.0.x runs /pre_start.sh before sshd and jupyter
# come up; it does not run /post_start.sh). A no-op on plain Docker. The app is started in the
# background so /start.sh can continue; logs go to the container's stdout and /var/log/groove-start.log.
groove-start > >(tee -a /var/log/groove-start.log) 2>&1 &
