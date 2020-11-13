#include <simple2d.h>

#include <cuda_runtime.h>
#include <iostream>
#include <string>

#define PI 3.14159265358979323846

S2D_Window* window;

float margins = 10;
float scaling = 50;
float weight = 2;
bool paused = false;
bool mousePressed = false;
float scrollSpeed = 0.05;

bool useCPU = false;

const double frequency = 10;

const double length = 1; // length in m, should be a whole number
const double ppm = 2048; // points per meter, should be a multiple of 1024
const double c = 1; // m/s

double dt = 0.00001;
int stepsPerFrame = 1000;

int nPoints = (int)(length * ppm);
double dx = length / nPoints;
double t = 0;

double* pos, * vel, * nPos;
double* d_pos, * d_vel, * d_nPos;
double* d_dx, * d_dt, * d_c;
int* d_N;

int NUM_THREADS = 1 << 10; // max (1024) threads per thread block
int NUM_BLOCKS = (nPoints + NUM_THREADS - 1) / NUM_THREADS;

int deviceId = -1;
bool GPUAllocated = false;

S2D_Text* txt = S2D_CreateText("LEMONMILK-Light.otf", "a", 15);

__global__ void timeStep(double* pos, double* vel, double* nPos, int* N, double* dx, double* dt, double* c, double input) {
	int TID = threadIdx.x + (blockIdx.x * blockDim.x);

	if (TID >= *N) {
		return;
	}

	if (TID == 0) {
		nPos[TID] = input;
		return;
	}

	double leftSlope = (pos[TID] - pos[TID - 1]) / (*dx);
	double rightSlope;
	if (TID == (*N) - 1) {
		rightSlope = leftSlope;
	}
	else {
		rightSlope = (pos[TID + 1] - pos[TID]) / (*dx);
	}

	double acc = pow((*c), 2.0) * (rightSlope - leftSlope) / (*dx);

	vel[TID] = vel[TID] + acc * (*dt);
	nPos[TID] = pos[TID] + vel[TID] * (*dt);
}

void timeStepCPU(int TID, double input) {
	if (TID >= nPoints) {
		return;
	}

	if (TID == 0) {
		nPos[TID] = input;
		return;
	}

	double leftSlope = (pos[TID] - pos[TID - 1]) / dx;
	double rightSlope;
	if (TID == (nPoints) - 1) {
		rightSlope = leftSlope;
	}
	else {
		rightSlope = (pos[TID + 1] - pos[TID]) / dx;
	}

	double acc = pow(c, 2.0) * (rightSlope - leftSlope) / dx;

	vel[TID] = vel[TID] + acc * dt;
	nPos[TID] = pos[TID] + vel[TID] * dt;
}

double wave(double time) {
	if (frequency * time > 20) return 0;
	return (-cos(2.0 * PI * frequency * time) + 1);
}

void sendToGPU() {
	if (useCPU) return;
	if (!GPUAllocated) {
		cudaMalloc(&d_pos, nPoints * sizeof(double));
		cudaMalloc(&d_vel, nPoints * sizeof(double));
		cudaMalloc(&d_nPos, nPoints * sizeof(double));
		cudaMalloc(&d_N, sizeof(int));
		cudaMalloc(&d_dx, sizeof(double));
		cudaMalloc(&d_dt, sizeof(double));
		cudaMalloc(&d_c, sizeof(double));
		GPUAllocated = true;
	}

	cudaMemcpy(d_pos, pos, nPoints * sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(d_vel, pos, nPoints * sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(d_N, &nPoints, sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_dx, &dx, sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(d_dt, &dt, sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(d_c, &c, sizeof(double), cudaMemcpyHostToDevice);
}

void swapPointers() {
	double* d_pos_temp, * pos_temp;

	// swap GPU pointers
	d_pos_temp = d_pos;
	d_pos = d_nPos;
	d_nPos = d_pos_temp;

	// swap CPU pointers
	pos_temp = pos;
	pos = nPos;
	nPos = pos_temp;
}

void reset() {
	for (int i = 0; i < nPoints; i++) {
		pos[i] = 0;
		vel[i] = 0;
		nPos[i] = 0;
	}
	t = 0;

	sendToGPU();
}

void setupVars() {
	pos = new double[nPoints];
	vel = new double[nPoints];
	nPos = new double[nPoints];
	reset();
}

std::string getDeviceName(int device) {
	cudaDeviceProp deviceProp;
	cudaGetDeviceProperties(&deviceProp, 0);
	return deviceProp.name;
}

void selectDevice() {
	int deviceCount;
	auto result = cudaGetDeviceCount(&deviceCount);
	if (deviceCount < 1 || result != cudaSuccess) {
		std::cout << "No CUDA enabled GPU's detected. Using CPU." << std::endl;
		useCPU = true;
		return;
	}

	if (deviceCount == 1) {
		std::cout << "Only one device detected, using that one." << std::endl;
		deviceId = 0;
		return;
	}
	std::cout << deviceCount << " compatible devices detected, please pick one: " << std::endl;

	while (deviceId < 0 || deviceId > deviceCount - 1) {
		for (int dev = 0; dev < deviceCount; dev++) {
			std::cout << "(" << dev << ") " << getDeviceName(dev) << std::endl;
		}

		std::string input = "";
		std::cin >> input;

		try {
			deviceId = std::stoi(input, nullptr, 10);
		}
		catch (...) {
			deviceId = -1;
		}

		if(deviceId < 0 || deviceId > deviceCount - 1) {
			std::cout << "Please enter a valid option." << std::endl;
		}
	}

	cudaSetDevice(deviceId);

}

void on_key(S2D_Event e) {
	if (e.type != S2D_KEY_DOWN) return;
	std::cout << e.key << " pressed" << std::endl;
	if (strcmp(e.key, "Space") == 0) {
		paused = !paused;
	}
	if (strcmp(e.key, "R") == 0) {
		reset();
	}
}

void on_mouse(S2D_Event e) {
	switch (e.type) {
	case S2D_MOUSE_SCROLL:
		int ddt = -scrollSpeed*(stepsPerFrame * e.delta_y + 1);
		if (ddt == 0) ddt = (e.delta_y < 0 ? 1 : -1);
		stepsPerFrame = stepsPerFrame + ddt;
		if (stepsPerFrame < 1) stepsPerFrame = 1;
		break;
	}
}

void update() {
	if (paused) {
		return;
	}

	int iteration = 0;
	while (iteration < stepsPerFrame) {
		double input = wave(t);
		if (useCPU) {
			for (int p = 0; p < nPoints; p++) {
				timeStepCPU(p, input);
			}
		}
		else {
			timeStep << <NUM_BLOCKS, NUM_THREADS >> > (d_pos, d_vel, d_nPos, d_N, d_dx, d_dt, d_c, input);
		}
		swapPointers();
		t += dt;
		iteration++;
	}
	if(!useCPU)
		cudaMemcpy(pos, d_pos, nPoints * sizeof(double), cudaMemcpyDeviceToHost);
}

void drawText(std::string text, float x, float y) {
	S2D_SetText(txt, text.c_str());
	txt->x = x;
	txt->y = y;
	S2D_DrawText(txt);
}

int currentLine = 0;
void printLine(std::string text) {
	drawText(text, 5, 5 + 15 * currentLine);
	currentLine++;
}

void render() {
	currentLine = 0;
	float y0 = window->viewport.height / 2;
	float x0 = margins;
	float spacing = (window->viewport.width - 2.0 * margins) / nPoints;
	for (int i = 0; i < nPoints - 1; i++) {
		S2D_DrawLine(
			x0 + spacing * i, y0 - scaling * pos[i],
			x0 + spacing * (i + 1), y0 - scaling * pos[i + 1],
			weight,
			1, 1, 1, 1,
			1, 1, 1, 1,
			1, 1, 1, 1,
			1, 1, 1, 1
		);
	}

	float timePerFrame = dt * stepsPerFrame;
	float simSpeed = timePerFrame * window->fps;

	printLine(paused ? "Paused" : "Running");
	printLine("Time: " + std::to_string(t));
	printLine("Steps per frame: " + std::to_string(stepsPerFrame));
	printLine("Step size: " + std::to_string(dt));
	printLine("Device: " + std::string(useCPU ? "CPU" : getDeviceName(deviceId)));
	printLine("FPS: " + std::to_string(window->fps));
	printLine("Simulation speed: " + std::to_string(simSpeed*100) + "% of realtime");
	printLine("");
	printLine("Controls:");
	printLine("Space: Pause simulation");
	printLine("R: Reset simulation");
	printLine("Scroll: Change simulation speed");
}

int main(int argc, char* argv[]) {
	selectDevice();
	setupVars();

	std::cout
		<< "Device: " << getDeviceName(deviceId) << std::endl
		<< "Blocks: " << NUM_BLOCKS << std::endl
		<< "Threads per block: " << NUM_THREADS << std::endl
		<< "Total threads: " << NUM_THREADS * NUM_BLOCKS << std::endl
		<< "Points: " << nPoints << std::endl;

	int inactiveThreads = NUM_THREADS * NUM_BLOCKS - nPoints;
	if (inactiveThreads) {
		std::cout << "WARNING: " << inactiveThreads << " inactive threads detected. Concider increasing nPoints to an integer multiple of 1024" << std::endl;
	}

	window = S2D_CreateWindow(
		"WaveSim",
		1280, 720,
		update, render,
		S2D_RESIZABLE
	);

	window->on_key = on_key;
	window->on_mouse = on_mouse;

	S2D_Show(window);

	return 0;

}