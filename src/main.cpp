#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <iostream>
#include <iomanip>
#include <sstream>
#include <cmath>
#include "world_manager.h"
#include "player.h"
#include "gl_renderer.h"

// Global state
static WorldManager g_world;
static Player g_player;
static GLRenderer g_renderer;

static bool g_mouse_look_active = true;
static bool g_right_mouse_down = false;
static bool g_first_mouse = true;
static double g_last_mouse_x = 0.0;
static double g_last_mouse_y = 0.0;
static float g_sim_time = 0.0f;

void cursor_position_callback(GLFWwindow* window, double xpos, double ypos) {
    if (g_first_mouse) {
        g_last_mouse_x = xpos;
        g_last_mouse_y = ypos;
        g_first_mouse = false;
        return;
    }

    double dx = xpos - g_last_mouse_x;
    double dy = ypos - g_last_mouse_y;
    g_last_mouse_x = xpos;
    g_last_mouse_y = ypos;

    // Reject discontinuous teleport/warp jumps (e.g., initial entry or recentering)
    if (std::abs(dx) > 120.0 || std::abs(dy) > 120.0) {
        return;
    }

    // Rotate camera if mouse look is active OR right mouse button is being dragged
    if (g_mouse_look_active || g_right_mouse_down) {
        const float sens = 0.0026f;
        // Un-inverted FPS mouse look:
        // dx > 0 (mouse right) -> yaw increases -> turn right
        // dy < 0 (mouse up in window space) -> -dy > 0 -> pitch increases -> look up
        g_player.rotate(static_cast<float>(dx * sens), static_cast<float>(-dy * sens));
    }

    // In mouse look mode, smoothly re-center cursor if near window border so user never runs out of screen
    if (g_mouse_look_active) {
        int win_w = 0, win_h = 0;
        glfwGetWindowSize(window, &win_w, &win_h);
        if (win_w > 0 && win_h > 0) {
            const double margin = 45.0;
            if (xpos < margin || xpos > (win_w - margin) ||
                ypos < margin || ypos > (win_h - margin)) {
                double cx = win_w * 0.5;
                double cy = win_h * 0.5;
                glfwSetCursorPos(window, cx, cy);
                g_last_mouse_x = cx;
                g_last_mouse_y = cy;
            }
        }
    }
}

void mouse_button_callback(GLFWwindow* window, int button, int action, int mods) {
    if (button == GLFW_MOUSE_BUTTON_LEFT && action == GLFW_PRESS) {
        // Fire ray gun and blast crater into terrain/objects!
        g_player.fireRayGun(g_world);
    } else if (button == GLFW_MOUSE_BUTTON_RIGHT) {
        if (action == GLFW_PRESS) {
            g_right_mouse_down = true;
        } else if (action == GLFW_RELEASE) {
            g_right_mouse_down = false;
        }
    }
}

void key_callback(GLFWwindow* window, int key, int scancode, int action, int mods) {
    if (action == GLFW_PRESS) {
        if (key == GLFW_KEY_ESCAPE) {
            glfwSetWindowShouldClose(window, GLFW_TRUE);
        } else if (key == GLFW_KEY_TAB || key == GLFW_KEY_M) {
            // Toggle mouse look mode
            g_mouse_look_active = !g_mouse_look_active;
            std::cout << "[Info] Mouse Look: " 
                      << (g_mouse_look_active ? "ENABLED (Move mouse to turn)" : "PAUSED (Right-drag or Arrow keys to turn)") 
                      << std::endl;
        } else if (key == GLFW_KEY_R) {
            // Respawn player in front of cabin
            g_player.init(0.0f, 9.0f, g_world);
            std::cout << "[Info] Player respawned at cabin clearing" << std::endl;
        }
    }
}

void framebuffer_size_callback(GLFWwindow* window, int width, int height) {
    if (width > 0 && height > 0) {
        g_renderer.resize(width, height);
    }
}

int main() {
    if (!glfwInit()) {
        std::cerr << "Failed to initialize GLFW!" << std::endl;
        return -1;
    }

    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

    int initial_width = 1280;
    int initial_height = 720;
    GLFWwindow* window = glfwCreateWindow(initial_width, initial_height, "CUDA Destructible SDF World - RTX 3090", nullptr, nullptr);
    if (!window) {
        std::cerr << "Failed to create GLFW window!" << std::endl;
        glfwTerminate();
        return -1;
    }

    glfwMakeContextCurrent(window);
    glfwSwapInterval(0); // Uncapped framerate

    // Use normal cursor mode to avoid WSLg/Xwayland relative pointer freeze
    glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_NORMAL);
    glfwSetCursorPos(window, initial_width * 0.5, initial_height * 0.5);
    g_last_mouse_x = initial_width * 0.5;
    g_last_mouse_y = initial_height * 0.5;
    g_first_mouse = false;

    glewExperimental = GL_TRUE;
    GLenum err = glewInit();
    if (err != GLEW_OK) {
        std::cerr << "GLEW init warning/error: " << glewGetErrorString(err) << std::endl;
        glGetError();
    }

    // Callbacks
    glfwSetCursorPosCallback(window, cursor_position_callback);
    glfwSetMouseButtonCallback(window, mouse_button_callback);
    glfwSetKeyCallback(window, key_callback);
    glfwSetFramebufferSizeCallback(window, framebuffer_size_callback);

    // Initialize world and player
    g_world.init();
    g_player.init(0.0f, 9.0f, g_world); // Start 9m in front of the cabin facing cabin

    if (!g_renderer.init(initial_width, initial_height)) {
        std::cerr << "Failed to initialize Renderer!" << std::endl;
        glfwDestroyWindow(window);
        glfwTerminate();
        return -1;
    }

    std::cout << "==========================================================" << std::endl;
    std::cout << "   CUDA Destructible SDF Outdoor World (RTX 3090)         " << std::endl;
    std::cout << "==========================================================" << std::endl;
    std::cout << "Controls:" << std::endl;
    std::cout << "  [Mouse Move]          : First-Person Look (Un-inverted)" << std::endl;
    std::cout << "  [Right-Click Drag]    : Drag View / Rotate Camera" << std::endl;
    std::cout << "  [Arrow Keys / Q, E]   : Smooth Camera Turn & Pitch" << std::endl;
    std::cout << "  [Left Click]          : Fire Ray Gun & Carve SDF Craters" << std::endl;
    std::cout << "  [W / A / S / D]       : Walk Across Terrain" << std::endl;
    std::cout << "  [Left Shift]          : Sprint" << std::endl;
    std::cout << "  [Spacebar]            : Jump" << std::endl;
    std::cout << "  [Tab / M]             : Toggle Mouse Look Mode" << std::endl;
    std::cout << "  [R]                   : Respawn at Cabin" << std::endl;
    std::cout << "  [Esc]                 : Exit Application" << std::endl;
    std::cout << "==========================================================" << std::endl;

    double last_time = glfwGetTime();
    double fps_timer = last_time;
    int frame_count = 0;

    while (!glfwWindowShouldClose(window)) {
        glfwPollEvents();

        double current_time = glfwGetTime();
        float delta_time = static_cast<float>(current_time - last_time);
        last_time = current_time;

        g_sim_time += delta_time;

        // Poll movement keys
        float fwd = 0.0f, strafe = 0.0f;
        if (glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS) fwd += 1.0f;
        if (glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS) fwd -= 1.0f;
        if (glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS) strafe += 1.0f;
        if (glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS) strafe -= 1.0f;
        bool sprint = (glfwGetKey(window, GLFW_KEY_LEFT_SHIFT) == GLFW_PRESS);
        bool jump = (glfwGetKey(window, GLFW_KEY_SPACE) == GLFW_PRESS);

        // Keyboard camera turning (Arrow Keys + Q / E)
        float turn_speed = sprint ? 2.5f : 1.7f;
        float rot_yaw = 0.0f;
        float rot_pitch = 0.0f;
        if (glfwGetKey(window, GLFW_KEY_RIGHT) == GLFW_PRESS || glfwGetKey(window, GLFW_KEY_E) == GLFW_PRESS) {
            rot_yaw += turn_speed * delta_time;
        }
        if (glfwGetKey(window, GLFW_KEY_LEFT) == GLFW_PRESS || glfwGetKey(window, GLFW_KEY_Q) == GLFW_PRESS) {
            rot_yaw -= turn_speed * delta_time;
        }
        if (glfwGetKey(window, GLFW_KEY_UP) == GLFW_PRESS || glfwGetKey(window, GLFW_KEY_PAGE_UP) == GLFW_PRESS) {
            rot_pitch += turn_speed * delta_time;
        }
        if (glfwGetKey(window, GLFW_KEY_DOWN) == GLFW_PRESS || glfwGetKey(window, GLFW_KEY_PAGE_DOWN) == GLFW_PRESS) {
            rot_pitch -= turn_speed * delta_time;
        }
        if (rot_yaw != 0.0f || rot_pitch != 0.0f) {
            g_player.rotate(rot_yaw, rot_pitch);
        }

        g_player.setMovementInput(fwd, strafe, sprint, jump);
        g_player.update(delta_time, g_world);

        int fb_width, fb_height;
        glfwGetFramebufferSize(window, &fb_width, &fb_height);

        FrameParams params = g_player.getFrameParams(fb_width, fb_height, g_sim_time);
        WorldView dev_world = g_world.getDeviceWorldView();

        g_renderer.render(params, dev_world);

        glfwSwapBuffers(window);
        frame_count++;

        if (current_time - fps_timer >= 0.5) {
            double fps = frame_count / (current_time - fps_timer);
            double ms_per_frame = 1000.0 / fps;

            glm::vec3 p = g_player.getPosition();
            std::ostringstream title;
            title << "CUDA SDF World [RTX 3090] | "
                  << std::fixed << std::setprecision(1) << fps << " FPS ("
                  << std::setprecision(2) << ms_per_frame << " ms) | "
                  << "Carves: " << g_world.getCarveCount() << " | "
                  << "Pos: (" << std::setprecision(1) << p.x << ", " << p.y << ", " << p.z << ")";
            glfwSetWindowTitle(window, title.str().c_str());

            frame_count = 0;
            fps_timer = current_time;
        }
    }

    g_renderer.cleanup();
    g_world.cleanup();
    glfwDestroyWindow(window);
    glfwTerminate();
    return 0;
}
