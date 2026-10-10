#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <iostream>
#include <iomanip>
#include <sstream>
#include "camera.h"
#include "gl_renderer.h"

// Global state
static OrbitCamera g_camera(4.5f, 0.8f, 0.35f);
static bool g_left_mouse_down = false;
static bool g_right_mouse_down = false;
static double g_last_cursor_x = 0.0;
static double g_last_cursor_y = 0.0;
static int g_scene_id = 1;
static bool g_paused = false;
static float g_sim_time = 0.0f;
static GLRenderer g_renderer;

void mouse_button_callback(GLFWwindow* window, int button, int action, int mods) {
    if (button == GLFW_MOUSE_BUTTON_LEFT) {
        g_left_mouse_down = (action == GLFW_PRESS);
        glfwGetCursorPos(window, &g_last_cursor_x, &g_last_cursor_y);
    } else if (button == GLFW_MOUSE_BUTTON_RIGHT) {
        g_right_mouse_down = (action == GLFW_PRESS);
        glfwGetCursorPos(window, &g_last_cursor_x, &g_last_cursor_y);
    }
}

void cursor_position_callback(GLFWwindow* window, double xpos, double ypos) {
    double dx = xpos - g_last_cursor_x;
    double dy = ypos - g_last_cursor_y;
    g_last_cursor_x = xpos;
    g_last_cursor_y = ypos;

    if (g_left_mouse_down) {
        g_camera.rotate(static_cast<float>(-dx * 0.005), static_cast<float>(-dy * 0.005));
    } else if (g_right_mouse_down) {
        g_camera.pan(static_cast<float>(dx), static_cast<float>(dy));
    }
}

void scroll_callback(GLFWwindow* window, double xoffset, double yoffset) {
    if (yoffset > 0) {
        g_camera.zoom(0.9f);
    } else if (yoffset < 0) {
        g_camera.zoom(1.1f);
    }
}

void key_callback(GLFWwindow* window, int key, int scancode, int action, int mods) {
    if (action == GLFW_PRESS) {
        if (key == GLFW_KEY_ESCAPE) {
            glfwSetWindowShouldClose(window, GLFW_TRUE);
        } else if (key == GLFW_KEY_1) {
            g_scene_id = 1;
            std::cout << "Switched to Scene 1: Smooth Morphing CSG" << std::endl;
        } else if (key == GLFW_KEY_2) {
            g_scene_id = 2;
            std::cout << "Switched to Scene 2: Mechanical CSG" << std::endl;
        } else if (key == GLFW_KEY_3) {
            g_scene_id = 3;
            std::cout << "Switched to Scene 3: Mathematical Gyroid" << std::endl;
        } else if (key == GLFW_KEY_SPACE) {
            g_paused = !g_paused;
            std::cout << (g_paused ? "Paused" : "Resumed") << std::endl;
        } else if (key == GLFW_KEY_R) {
            g_camera.reset();
            std::cout << "Camera Reset" << std::endl;
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
    GLFWwindow* window = glfwCreateWindow(initial_width, initial_height, "CUDA SDF Raymarcher - RTX 3090", nullptr, nullptr);
    if (!window) {
        std::cerr << "Failed to create GLFW window!" << std::endl;
        glfwTerminate();
        return -1;
    }

    glfwMakeContextCurrent(window);
    glfwSwapInterval(0); // Uncapped framerate

    glewExperimental = GL_TRUE;
    GLenum err = glewInit();
    if (err != GLEW_OK) {
        std::cerr << "GLEW init warning/error: " << glewGetErrorString(err) << std::endl;
        // In modern OpenGL core profiles, glewInit() sometimes sets GL_INVALID_ENUM on glGetString(GL_EXTENSIONS). Clear it:
        glGetError();
    }

    std::cout << "OpenGL Vendor: " << glGetString(GL_VENDOR) << std::endl;
    std::cout << "OpenGL Renderer: " << glGetString(GL_RENDERER) << std::endl;
    std::cout << "OpenGL Version: " << glGetString(GL_VERSION) << std::endl;

    // Callbacks
    glfwSetMouseButtonCallback(window, mouse_button_callback);
    glfwSetCursorPosCallback(window, cursor_position_callback);
    glfwSetScrollCallback(window, scroll_callback);
    glfwSetKeyCallback(window, key_callback);
    glfwSetFramebufferSizeCallback(window, framebuffer_size_callback);

    if (!g_renderer.init(initial_width, initial_height)) {
        std::cerr << "Failed to initialize Renderer!" << std::endl;
        glfwDestroyWindow(window);
        glfwTerminate();
        return -1;
    }

    std::cout << "===================================================" << std::endl;
    std::cout << "   CUDA SDF Real-Time Raymarcher (RTX 3090)        " << std::endl;
    std::cout << "===================================================" << std::endl;
    std::cout << "Controls:" << std::endl;
    std::cout << "  [Left Mouse Drag]   : Orbit Rotate" << std::endl;
    std::cout << "  [Right Mouse Drag]  : Pan Camera" << std::endl;
    std::cout << "  [Scroll Wheel]      : Zoom In / Out" << std::endl;
    std::cout << "  [1, 2, 3 Keys]      : Switch SDF Scenes" << std::endl;
    std::cout << "  [Space]             : Pause / Resume Animation" << std::endl;
    std::cout << "  [R]                 : Reset Camera" << std::endl;
    std::cout << "  [Esc]               : Exit" << std::endl;
    std::cout << "===================================================" << std::endl;

    double last_time = glfwGetTime();
    double fps_timer = last_time;
    int frame_count = 0;

    while (!glfwWindowShouldClose(window)) {
        glfwPollEvents();

        double current_time = glfwGetTime();
        float delta_time = static_cast<float>(current_time - last_time);
        last_time = current_time;

        if (!g_paused) {
            g_sim_time += delta_time;
        }

        int fb_width, fb_height;
        glfwGetFramebufferSize(window, &fb_width, &fb_height);

        CameraUniforms cam_uniforms = g_camera.getUniforms(fb_width, fb_height);
        g_renderer.render(cam_uniforms, g_sim_time, g_scene_id);

        glfwSwapBuffers(window);
        frame_count++;

        if (current_time - fps_timer >= 0.5) {
            double fps = frame_count / (current_time - fps_timer);
            double ms_per_frame = 1000.0 / fps;

            std::ostringstream title;
            title << "CUDA SDF Raymarcher [RTX 3090] | "
                  << std::fixed << std::setprecision(1) << fps << " FPS ("
                  << std::setprecision(2) << ms_per_frame << " ms) | Scene " << g_scene_id;
            glfwSetWindowTitle(window, title.str().c_str());

            frame_count = 0;
            fps_timer = current_time;
        }
    }

    g_renderer.cleanup();
    glfwDestroyWindow(window);
    glfwTerminate();
    return 0;
}
