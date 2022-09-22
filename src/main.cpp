/*********************************************************************
 * Software License Agreement (BSD License)
 *
 *  Copyright (c) 2022, PickNik Inc
 *  All rights reserved.
 *
 *  Redistribution and use in source and binary forms, with or without
 *  modification, are permitted provided that the following conditions
 *  are met:
 *
 *   * Redistributions of source code must retain the above copyright
 *     notice, this list of conditions and the following disclaimer.
 *   * Redistributions in binary form must reproduce the above
 *     copyright notice, this list of conditions and the following
 *     disclaimer in the documentation and/or other materials provided
 *     with the distribution.
 *   * Neither the name of PickNik Inc nor the names of its
 *     contributors may be used to endorse or promote products derived
 *     from this software without specific prior written permission.
 *
 *  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 *  "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 *  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
 *  FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
 *  COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
 *  INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
 *  BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 *  LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 *  CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
 *  LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
 *  ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 *  POSSIBILITY OF SUCH DAMAGE.
 *********************************************************************/

#include <chrono>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <memory>
#include <mutex>
#include <string>
#include <thread>

// ROS
#include <mujocoros2/mujocoros2.hpp>
#include <rclcpp/executors/single_threaded_executor.hpp>
#include <rclcpp/logging.hpp>
#include <rclcpp/rclcpp.hpp>

// #include <mujoco/simulate/simulate.h>
// #include <mujoco/simulate/array_safety.h>
// #include <mujoco/simulate/glfw_dispatch.h>
#include <mujoco/mujoco.h>

// int main(int argc, char ** argv)
// {
//   // Initialize ROS
//   rclcpp::init(argc, argv);
//   rclcpp::Logger logger = rclcpp::get_logger("mujocoros2");
//   RCLCPP_INFO(logger, "Starting mujocoros2...");

//   // Create a node.
//   auto node = std::make_shared<mujocoros2::mujocoros2>();

//   /* Create an executor that will be responsible for execution of callbacks for
//    * a set of nodes. With this version, all callbacks will be called from within
//    * this thread (the main one). */
//   rclcpp::executors::SingleThreadedExecutor exec;
//   exec.add_node(node);

//   /* spin will block until work comes in, execute work as it becomes available,
//    * and keep blocking. It will only be interrupted by Ctrl-C. */
//   exec.spin();
//   rclcpp::shutdown();
//   return 0;
// }

amespace
{
  namespace mj = ::mujoco;
  namespace mju = ::mujoco::sample_util;

  using ::mujoco::Glfw;

  // constants
  const double syncMisalign = 0.1;        // maximum misalignment before re-sync (simulation seconds)
  const double simRefreshFraction = 0.7;  // fraction of refresh available for simulation
  const int kErrorLength = 1024;          // load error string length

  // model and data
  mjModel* m = nullptr;
  mjData* d = nullptr;

  // control noise variables
  mjtNum* ctrlnoise = nullptr;

  //------------------------------------------- simulation -------------------------------------------

  mjModel* LoadModel(const char* file, mj::Simulate& sim)
  {
    // this copy is needed so that the mju::strlen call below compiles
    char filename[mj::Simulate::kMaxFilenameLength];
    mju::strcpy_arr(filename, file);

    // make sure filename is not empty
    if (!filename[0])
    {
      return nullptr;
    }

    // load and compile
    char loadError[kErrorLength] = "";
    mjModel* mnew = 0;
    if (mju::strlen_arr(filename) > 4 && !std::strncmp(filename + mju::strlen_arr(filename) - 4, ".mjb",
                                                       mju::sizeof_arr(filename) - mju::strlen_arr(filename) + 4))
    {
      mnew = mj_loadModel(filename, nullptr);
      if (!mnew)
      {
        mju::strcpy_arr(loadError, "could not load binary model");
      }
    }
    else
    {
      mnew = mj_loadXML(filename, nullptr, loadError, mj::Simulate::kMaxFilenameLength);
      // remove trailing newline character from loadError
      if (loadError[0])
      {
        int error_length = mju::strlen_arr(loadError);
        if (loadError[error_length - 1] == '\n')
        {
          loadError[error_length - 1] = '\0';
        }
      }
    }

    mju::strcpy_arr(sim.loadError, loadError);

    if (!mnew)
    {
      std::printf("%s\n", loadError);
      return nullptr;
    }

    // compiler warning: print and pause
    if (loadError[0])
    {
      // mj_forward() below will print the warning message
      std::printf("Model compiled, but simulation warning (paused):\n  %s\n", loadError);
      sim.run = 0;
    }

    return mnew;
  }

  // simulate in background thread (while rendering in main thread)
  void PhysicsLoop(mj::Simulate & sim)
  {
    // cpu-sim synchronization point
    double syncCPU = 0;
    mjtNum syncSim = 0;

    // run until asked to exit
    while (!sim.exitrequest.load())
    {
      if (sim.droploadrequest.load())
      {
        mjModel* mnew = LoadModel(sim.dropfilename, sim);
        sim.droploadrequest.store(false);

        mjData* dnew = nullptr;
        if (mnew)
          dnew = mj_makeData(mnew);
        if (dnew)
        {
          sim.load(sim.dropfilename, mnew, dnew, true);

          m = mnew;
          d = dnew;
          mj_forward(m, d);

          // allocate ctrlnoise
          free(ctrlnoise);
          ctrlnoise = (mjtNum*)malloc(sizeof(mjtNum) * m->nu);
          mju_zero(ctrlnoise, m->nu);
        }
      }

      if (sim.uiloadrequest.load())
      {
        sim.uiloadrequest.fetch_sub(1);
        mjModel* mnew = LoadModel(sim.filename, sim);
        mjData* dnew = nullptr;
        if (mnew)
          dnew = mj_makeData(mnew);
        if (dnew)
        {
          sim.load(sim.filename, mnew, dnew, true);

          m = mnew;
          d = dnew;
          mj_forward(m, d);

          // allocate ctrlnoise
          free(ctrlnoise);
          ctrlnoise = static_cast<mjtNum*>(malloc(sizeof(mjtNum) * m->nu));
          mju_zero(ctrlnoise, m->nu);
        }
      }

      // sleep for 1 ms or yield, to let main thread run
      //  yield results in busy wait - which has better timing but kills battery life
      if (sim.run && sim.busywait)
      {
        std::this_thread::yield();
      }
      else
      {
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
      }

      {
        // lock the sim mutex
        const std::lock_guard<std::mutex> lock(sim.mtx);

        // run only if model is present
        if (m)
        {
          // running
          if (sim.run)
          {
            // record cpu time at start of iteration
            double startCPU = Glfw().glfwGetTime();

            // elapsed CPU and simulation time since last sync
            double elapsedCPU = startCPU - syncCPU;
            double elapsedSim = d->time - syncSim;

            // inject noise
            if (sim.ctrlnoisestd)
            {
              // convert rate and scale to discrete time (Ornstein–Uhlenbeck)
              mjtNum rate = mju_exp(-m->opt.timestep / sim.ctrlnoiserate);
              mjtNum scale = sim.ctrlnoisestd * mju_sqrt(1 - rate * rate);

              for (int i = 0; i < m->nu; i++)
              {
                // update noise
                ctrlnoise[i] = rate * ctrlnoise[i] + scale * mju_standardNormal(nullptr);

                // apply noise
                d->ctrl[i] = ctrlnoise[i];
              }
            }

            // requested slow-down factor
            double slowdown = 100 / sim.percentRealTime[sim.realTimeIndex];

            // misalignment condition: distance from target sim time is bigger than syncmisalign
            bool misaligned = mju_abs(elapsedCPU / slowdown - elapsedSim) > syncMisalign;

            // out-of-sync (for any reason): reset sync times, step
            if (elapsedSim < 0 || elapsedCPU < 0 || syncCPU == 0 || misaligned || sim.speedChanged)
            {
              // re-sync
              syncCPU = startCPU;
              syncSim = d->time;
              sim.speedChanged = false;

              // clear old perturbations, apply new
              mju_zero(d->xfrc_applied, 6 * m->nbody);
              sim.applyposepertubations(0);  // move mocap bodies only
              sim.applyforceperturbations();

              // run single step, let next iteration deal with timing
              mj_step(m, d);
            }

            // in-sync: step until ahead of cpu
            else
            {
              bool measured = false;
              mjtNum prevSim = d->time;
              double refreshTime = simRefreshFraction / sim.refreshRate;

              // step while sim lags behind cpu and within refreshTime
              while ((d->time - syncSim) * slowdown < (Glfw().glfwGetTime() - syncCPU) &&
                     (Glfw().glfwGetTime() - startCPU) < refreshTime)
              {
                // measure slowdown before first step
                if (!measured && elapsedSim)
                {
                  sim.measuredSlowdown = elapsedCPU / elapsedSim;
                  measured = true;
                }

                // clear old perturbations, apply new
                mju_zero(d->xfrc_applied, 6 * m->nbody);
                sim.applyposepertubations(0);  // move mocap bodies only
                sim.applyforceperturbations();

                // call mj_step
                mj_step(m, d);

                // break if reset
                if (d->time < prevSim)
                {
                  break;
                }
              }
            }
          }

          // paused
          else
          {
            // apply pose perturbation
            sim.applyposepertubations(1);  // move mocap and dynamic bodies

            // run mj_forward, to update rendering and joint sliders
            mj_forward(m, d);
          }
        }
      }  // release std::lock_guard<std::mutex>
    }
  }
}  // namespace

//-------------------------------------- physics_thread --------------------------------------------

void PhysicsThread(mj::Simulate* sim, const char* filename)
{
  // request loadmodel if file given (otherwise drag-and-drop)
  if (filename != nullptr)
  {
    m = LoadModel(filename, *sim);
    if (m)
      d = mj_makeData(m);
    if (d)
    {
      sim->load(filename, m, d, true);
      mj_forward(m, d);

      // allocate ctrlnoise
      free(ctrlnoise);
      ctrlnoise = static_cast<mjtNum*>(malloc(sizeof(mjtNum) * m->nu));
      mju_zero(ctrlnoise, m->nu);
    }
  }

  PhysicsLoop(*sim);

  // delete everything we allocated
  free(ctrlnoise);
  mj_deleteData(d);
  mj_deleteModel(m);
}

//------------------------------------------ main --------------------------------------------------

// run event loop
int main(int argc, const char** argv)
{
  // print version, check compatibility
  std::printf("MuJoCo version %s\n", mj_versionString());
  if (mjVERSION_HEADER != mj_version())
  {
    mju_error("Headers and library have different versions");
  }

  // simulate object encapsulates the UI
  auto sim = std::make_unique<mj::Simulate>();

  // init GLFW
  if (!Glfw().glfwInit())
  {
    mju_error("could not initialize GLFW");
  }

  const char* filename = nullptr;
  if (argc > 1)
  {
    filename = argv[1];
  }

  // start physics thread
  std::thread physicsthreadhandle = std::thread(&PhysicsThread, sim.get(), filename);

  // start simulation UI loop (blocking call)
  sim->renderloop();
  physicsthreadhandle.join();

  // terminate GLFW (crashes with Linux NVidia drivers)
#if defined(__APPLE__) || defined(_WIN32)
  Glfw().glfwTerminate();
#endif

  return 0;
}
