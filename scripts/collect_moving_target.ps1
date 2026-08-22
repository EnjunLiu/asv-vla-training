$ErrorActionPreference = "Stop"

$sshTarget = "jetson@192.168.137.100"
$repo = "D:\asv-vla-training"
$episodeRoot = Join-Path $repo "data\episodes\moving_target"
$remoteRoot = "/tmp/asv_moving_collection_20260820"
$ue = "D:\Softwares\Unreal Engine\UE_5.6\Engine\Binaries\Win64\UnrealEditor.exe"
$project = "D:\asv-unreal-simulation\VLA.uproject"

function Invoke-Remote([string]$Command) {
    & ssh -o BatchMode=yes $sshTarget $Command
    if ($LASTEXITCODE -ne 0) {
        throw "remote command failed with exit code $LASTEXITCODE"
    }
}

if (Test-Path $episodeRoot) {
    $existing = Get-ChildItem -LiteralPath $episodeRoot -Directory -ErrorAction SilentlyContinue
    if ($existing) {
        throw "refusing to overwrite existing moving-target episodes"
    }
}
New-Item -ItemType Directory -Force -Path $episodeRoot | Out-Null

$scenarios = @(
    @{ Slot = "RED_3M_TRAIN_01"; Color = "red"; Distance = 3; Layout = "L7"; Seed = 231101 },
    @{ Slot = "RED_3M_TRAIN_02"; Color = "red"; Distance = 3; Layout = "L7B"; Seed = 231102 },
    @{ Slot = "RED_3M_TRAIN_03"; Color = "red"; Distance = 3; Layout = "L7"; Seed = 231103 },
    @{ Slot = "RED_3M_TRAIN_04"; Color = "red"; Distance = 3; Layout = "L7B"; Seed = 231104 },
    @{ Slot = "RED_3M_VALIDATION"; Color = "red"; Distance = 3; Layout = "L7"; Seed = 231105 },
    @{ Slot = "RED_3M_TEST"; Color = "red"; Distance = 3; Layout = "L7B"; Seed = 231106 },
    @{ Slot = "BLUE_3M_TRAIN_01"; Color = "blue"; Distance = 3; Layout = "L7"; Seed = 231201 },
    @{ Slot = "BLUE_3M_TRAIN_02"; Color = "blue"; Distance = 3; Layout = "L7B"; Seed = 231202 },
    @{ Slot = "BLUE_3M_TRAIN_03"; Color = "blue"; Distance = 3; Layout = "L7"; Seed = 231203 },
    @{ Slot = "BLUE_3M_TRAIN_04"; Color = "blue"; Distance = 3; Layout = "L7B"; Seed = 231204 },
    @{ Slot = "BLUE_3M_VALIDATION"; Color = "blue"; Distance = 3; Layout = "L7"; Seed = 231205 },
    @{ Slot = "BLUE_3M_TEST"; Color = "blue"; Distance = 3; Layout = "L7B"; Seed = 231206 },
    @{ Slot = "RED_4M_TRAIN_01"; Color = "red"; Distance = 4; Layout = "L7"; Seed = 231301 },
    @{ Slot = "RED_4M_TRAIN_02"; Color = "red"; Distance = 4; Layout = "L7B"; Seed = 231302 },
    @{ Slot = "RED_4M_TRAIN_03"; Color = "red"; Distance = 4; Layout = "L7"; Seed = 231303 },
    @{ Slot = "RED_4M_TRAIN_04"; Color = "red"; Distance = 4; Layout = "L7B"; Seed = 231304 },
    @{ Slot = "RED_4M_VALIDATION"; Color = "red"; Distance = 4; Layout = "L7"; Seed = 231305 },
    @{ Slot = "RED_4M_TEST"; Color = "red"; Distance = 4; Layout = "L7B"; Seed = 231306 }
)

Invoke-Remote "set -e; test ! -e $remoteRoot; mkdir -p $remoteRoot; cd /home/jetson/jetson_asv_ws; source /opt/ros/humble/setup.bash; source install/setup.bash; nohup ros2 run bridge bridge_node --ros-args --params-file src/bridge/config/ue_bridge.yaml -p outbound_command_mode:=disabled -p use_sim_time:=true > $remoteRoot/bridge.log 2>&1 < /dev/null & echo `$! > $remoteRoot/bridge.pid; sleep 3"

try {
    foreach ($scenario in $scenarios) {
        $slot = $scenario.Slot
        $color = $scenario.Color
        $distance = $scenario.Distance
        $layout = $scenario.Layout
        $seed = $scenario.Seed
        $remoteEpisode = "$remoteRoot/$slot"
        $task = "follow the $color boat, keep $distance meters distance"
        $archive = "$remoteRoot/$slot.tar.gz"
        $localArchive = Join-Path $repo "data\raw\$slot.tar.gz"
        $localEpisode = Join-Path $episodeRoot $slot

        Write-Host "COLLECT_START $slot seed=$seed"
        Invoke-Remote "set -e; mkdir -p $remoteEpisode; cd /home/jetson/jetson_asv_ws; source /opt/ros/humble/setup.bash; source install/setup.bash; nohup ros2 bag record -o $remoteEpisode/bag /ue/asv_state /ue/camera_frame /ue/entities > $remoteEpisode/bag.log 2>&1 < /dev/null & echo `$! > $remoteEpisode/bag.pid; sleep 3"

        $args = @(
            $project, "Main_Map", "-game", "-SceneAuto", "-unattended", "-RenderOffscreen", "-nosplash",
            "-benchmark", "-fps=10", "-Slot=$slot", "-Layout=$layout", "-Motion=S2", "-Seed=$seed",
            "-MaxRuntimeSeconds=90", "-RenderWarmupSeconds=1", "-SineDelay=0",
            "-ExpertFollowColor=$color", "-ExpertStandoffM=$distance", "-ExpertMaxStepCm=30",
            "-ExpertMaxAccelerationCmPerSec2=1200", "-log"
        )
        $process = Start-Process -FilePath $ue -ArgumentList $args -WindowStyle Hidden -PassThru -Wait
        if ($process.ExitCode -ne 0) {
            throw "UE5 failed for $slot with exit code $($process.ExitCode)"
        }

        Invoke-Remote "set -e; bag_pid=`$(cat $remoteEpisode/bag.pid); kill -INT `$bag_pid; sleep 4; cd /home/jetson/jetson_asv_ws; source /opt/ros/humble/setup.bash; source install/setup.bash; python3 /tmp/export_moving_target_bag.py --bag $remoteEpisode/bag --output $remoteEpisode/episode --task '$task' --slot $slot --layout $layout --motion S2; tar -czf $archive -C $remoteEpisode episode"
        New-Item -ItemType Directory -Force -Path (Split-Path $localArchive) | Out-Null
        & scp -q "$sshTarget`:$archive" $localArchive
        if ($LASTEXITCODE -ne 0) { throw "scp failed for $slot" }
        New-Item -ItemType Directory -Force -Path $localEpisode | Out-Null
        & tar -xzf $localArchive -C $localEpisode --strip-components=1
        if ($LASTEXITCODE -ne 0) { throw "local extraction failed for $slot" }
        $manifest = Get-Content (Join-Path $localEpisode "manifest.json") -Raw | ConvertFrom-Json
        Write-Host "COLLECT_PASS $slot frames=$($manifest.frame_count) run_id=$($manifest.run_id)"
    }
}
finally {
    Invoke-Remote "set +e; if test -f $remoteRoot/bridge.pid; then kill -INT `$(cat $remoteRoot/bridge.pid) 2>/dev/null; fi"
}

Write-Host "MOVING_TARGET_COLLECTION_COMPLETE root=$episodeRoot"
