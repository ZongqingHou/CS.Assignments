//******************************************************************************
#include "stm32f4xx.h"
#include "stm32f4xx_gpio.h"
#include "stm32f4xx_rcc.h"
#include "stm32f4xx_conf.h"
#include "discoveryf4utils.h"
//******************************************************************************
#include "FreeRTOS.h"
#include "task.h"
#include "semphr.h"
//******************************************************************************
#include "codec.h"
//******************************************************************************
#define NOTEFREQUENCY 0.015
#define NOTEAMPLITUDE 500.0
//******************************************************************************
#define STACK_SIZE_MIN	128
//******************************************************************************
typedef struct {
	float tabs[8];
	float params[8];
	uint8_t currIndex;
} fir_8;
//******************************************************************************
typedef enum {
	run = 0,
	wait = 1,
	ready = 2,
	finish = 3
} coffee_state;

typedef struct {
	uint8_t state;
	uint8_t brewing_dutation;
	uint8_t deadline;
	uint16_t period;
	uint8_t priority;
	uint8_t running;
	uint8_t led_index;
	uint8_t list_index;
	uint8_t used;
	
// TimerHandle_t BrewCoffee;
	xSemaphoreHandle xCoffeeSemaphore;
} coffee_setting;
//******************************************************************************
volatile uint32_t sampleCounter = 0;
volatile int16_t sample = 0;

double sawWave = 0.0;

float filteredSaw = 0.0;
//******************************************************************************
fir_8 filt;

uint8_t glb_index;

uint8_t LED_FLAG = 0;
uint8_t MENU_FLAG = 1;
uint8_t RUNNING_FLAG = 0;
uint8_t SAME_TIME_FLAG = 1;
uint8_t START_FLAG = 0;
uint8_t space_flag = 0;
uint8_t brewing_count = 0;
uint8_t task_count = 0;

uint32_t sound_period = 0;
uint32_t time_count = 0;

coffee_setting brewing_list[4];

xSemaphoreHandle xButtonSemaphore;
xSemaphoreHandle xSoundSemaphore;
xSemaphoreHandle xStartSemaphore;

BaseType_t * sHigherPriorityTaskWoken;

TaskHandle_t xHandleBrewing[4];
//******************************************************************************
void sound(void);
float updateFilter(fir_8* theFilter, float newValue);
void initFilter(fir_8* theFilter);
//******************************************************************************
void vLedTask(void *pvParameters);
void vButtonTask(void *pvParameters);
void vSoundTask(void *pvParameters);
//******************************************************************************
void unsetLed(void);
void setLed(void);
void setList(void);
void initCof(coffee_setting* cof, int led_index, int list_index);
void sUsedFlag(void);
void vFPS(void *pvParameters);
void vEDF(void *pvParameters);
void vLLS(void *pvParameters);
