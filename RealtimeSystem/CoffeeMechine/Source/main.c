//******************************************************************************
#include "main.h"
//******************************************************************************
int
main(void)
{
	NVIC_PriorityGroupConfig( NVIC_PriorityGroup_4 );

	STM_EVAL_LEDInit(LED_BLUE);
	STM_EVAL_LEDInit(LED_GREEN);
	STM_EVAL_LEDInit(LED_ORANGE);
	STM_EVAL_LEDInit(LED_RED);

	STM_EVAL_PBInit(BUTTON_USER, BUTTON_MODE_GPIO);

	RCC_AHB1PeriphClockCmd(RCC_AHB1Periph_GPIOC, ENABLE);

	codec_init();
	codec_ctrl_init();

	I2S_Cmd(CODEC_I2S, ENABLE);

	initFilter(&filt);

	vSemaphoreCreateBinary(xButtonSemaphore);
	xSemaphoreTake(xButtonSemaphore, 0);

	vSemaphoreCreateBinary(xSoundSemaphore);
	xSemaphoreTake(xSoundSemaphore, 0);

	vSemaphoreCreateBinary(xStartSemaphore);
	xSemaphoreTake(xStartSemaphore, 0);

	if (xButtonSemaphore != NULL && xSoundSemaphore != NULL && xStartSemaphore != NULL) {
		xTaskCreate(vLedTask, (const char * )"LED", STACK_SIZE_MIN, NULL, tskIDLE_PRIORITY, NULL);
		xTaskCreate(vButtonTask, (const char * )"BUTTON", STACK_SIZE_MIN, NULL, tskIDLE_PRIORITY, NULL);
		xTaskCreate(vSoundTask, (const char * )"SOUND", STACK_SIZE_MIN, NULL, tskIDLE_PRIORITY, NULL);
	}

	vTaskStartScheduler();
}
//******************************************************************************
void
vLedTask(void *pvParameters)
{
	for (;;)
	{
///*
//-------------------------------------------------------------------------------------------
		// user menu
		if (MENU_FLAG) {
			setLed();
			time_count = 0;
		} else {
			if (xSemaphoreTakeFromISR(xButtonSemaphore, sHigherPriorityTaskWoken)) {
				vTaskDelay(500 / portTICK_RATE_MS);
				unsetLed();

				RUNNING_FLAG = 1;
				for (;;) {
					if (RUNNING_FLAG) {

						// create 4 tasks and make it start at the same time
						for (glb_index = 0; glb_index < 4; glb_index++) {
							initCof(&brewing_list[glb_index], glb_index, glb_index);
							brewing_count++;
							task_count++;
							xTaskCreate(vFPS, (const char * )"vFPS", STACK_SIZE_MIN, &brewing_list[glb_index], tskIDLE_PRIORITY, &xHandleBrewing[glb_index]);
							//xTaskCreate(vEDF, (const char * )"vEDF", STACK_SIZE_MIN, &brewing_list[glb_index], tskIDLE_PRIORITY, &xHandleBrewing[glb_index]);
							//xTaskCreate(vLLS, (const char * )"vLLS", STACK_SIZE_MIN, &brewing_list[glb_index], tskIDLE_PRIORITY, &xHandleBrewing[glb_index]);
						}
						RUNNING_FLAG = 0;
						START_FLAG = 1;
						xSemaphoreGive(xStartSemaphore);
						xTaskCreate(vTimerTask, (const char * )"vFPS", STACK_SIZE_MIN, NULL, tskIDLE_PRIORITY, NULL);
					}

					if (task_count == 0 && !RUNNING_FLAG) {
						RUNNING_FLAG = 1;
					}
				}
			}
		}
//*/
		/*
		//-------------------------------------------------------------------------------------------
				// coffee selection
				if (MENU_FLAG) {
					setLed();
					setList();
					time_count = 0;
				} else {
					if (xSemaphoreTakeFromISR(xButtonSemaphore, sHigherPriorityTaskWoken)) {
						vTaskDelay(500 / portTICK_RATE_MS);
						unsetLed();

						// double press button check
						if (xSemaphoreTakeFromISR(xButtonSemaphore, sHigherPriorityTaskWoken)) {
							for(glb_index = 0; glb_index < 3; glb_index++){
								if(brewing_list[glb_index].state == finish){
									space_flag = 1;
									break;
								}
							}

							if(space_flag){
								if(task_count == 0)
								{
									START_FLAG = 1;
									time_count = 0;
								}
								initCof(&brewing_list[glb_index], LED_FLAG - 1, glb_index);
								xTaskCreate(vFPS, (const char * )"vFPS", STACK_SIZE_MIN, &brewing_list[glb_index], tskIDLE_PRIORITY, &xHandleBrewing[glb_index]);
								//xTaskCreate(vEDF, (const char * )"vEDF", STACK_SIZE_MIN, &brewing_list[glb_index], tskIDLE_PRIORITY, &xHandleBrewing[glb_index]);
								//xTaskCreate(vLLS, (const char * )"vLLS", STACK_SIZE_MIN, &brewing_list[glb_index], tskIDLE_PRIORITY, &xHandleBrewing[glb_index]);
								task_count++;
								if(brewing_count < 3){
									brewing_count++;
								}
								xSemaphoreGive(xStartSemaphore);
							}
						} else {
								STM_EVAL_LEDOn(LED_FLAG);
								LED_FLAG = (LED_FLAG + 1) % 4;
						}
					}
				}
		//-------------------------------------------------------------------------------------------
		*/
	}
}
//******************************************************************************
void
vButtonTask(void *pvParameters)
{
	for (;;)
	{
		if (STM_EVAL_PBGetState(BUTTON_USER)) {
			xSemaphoreGiveFromISR(xButtonSemaphore, sHigherPriorityTaskWoken );
			MENU_FLAG = 0;
			vTaskDelay(200 / portTICK_RATE_MS);
		}
	}
}
//******************************************************************************
void
vSoundTask(void *pvParameters)
{
	for (;;) {
		if (xSemaphoreTake(xSoundSemaphore, portMAX_DELAY)) {
			while (sound_period < 1000000) {
				sound();
				sound_period += 1;
			}
			codec_init();
			codec_ctrl_init();

			I2S_Cmd(CODEC_I2S, ENABLE);

			initFilter(&filt);
			sound_period = 0;
		}
	}
}
//******************************************************************************
void Brewing(TimerHandle_t xTimer)
{
	time_count++;
}
//******************************************************************************
void
vTimerTask(void *pvParameter)
{
	TimerHandle_t BrewCoffee;

	for (;;)
	{
		BrewCoffee = xTimerCreate((const char* )"Timer1", (TickType_t )1000, (UBaseType_t )pdTRUE, (void* )1, (TimerCallbackFunction_t)Brewing);
		xTimerStart(BrewCoffee, 0);
	}
}
//******************************************************************************
void
unsetLed()
{
	STM_EVAL_LEDOff(LED_BLUE);
	STM_EVAL_LEDOff(LED_RED);
	STM_EVAL_LEDOff(LED_GREEN);
	STM_EVAL_LEDOff(LED_ORANGE);
}

void
setLed()
{
	STM_EVAL_LEDOn(LED_BLUE);
	STM_EVAL_LEDOn(LED_ORANGE);
	STM_EVAL_LEDOn(LED_RED);
	STM_EVAL_LEDOn(LED_GREEN);
}

void
setList()
{
	for (glb_index = 0; glb_index < 3; glb_index++) {
		brewing_list[glb_index].state = finish;
	}
}
//******************************************************************************
void
initCof(coffee_setting* cof, int led_index, int list_index)
{
	switch (led_index) {
	case 1:
		cof->brewing_dutation = 3;
		cof->deadline = 5;
		cof->period = 20;
		cof->priority = 3;
		cof->led_index = led_index;
		break;
	case 0:
		cof->brewing_dutation = 4;
		cof->deadline = 10;
		cof->period = 30;
		cof->priority = 1;
		cof->led_index = led_index;
		break;
	case 2:
		cof->brewing_dutation = 6;
		cof->deadline = 15;
		cof->period = 40;
		cof->priority = 2;
		cof->led_index = led_index;
		break;
	default:
		cof->brewing_dutation = 4;
		cof->deadline = 10;
		cof->period = 40;
		cof->priority = 2;
		cof->led_index = 3;
		break;
	}

	vSemaphoreCreateBinary(cof->xCoffeeSemaphore);
	xSemaphoreTake(cof->xCoffeeSemaphore, 0);

	// for the observation
	//cof->period *= 100;

	cof->list_index = list_index;
	cof->state = wait;
	cof->used = 0;
	cof->running = 0;
}
//******************************************************************************
void
sound(void)
{
	if (SPI_I2S_GetFlagStatus(CODEC_I2S, SPI_I2S_FLAG_TXE))
	{
		SPI_I2S_SendData(CODEC_I2S, sample);
		if (sampleCounter & 0x00000001)
		{
			sawWave += NOTEFREQUENCY;
			if (sawWave > 1.0)
				sawWave -= 2.0;

			filteredSaw = updateFilter(&filt, sawWave);
			sample = (int16_t)(NOTEAMPLITUDE * filteredSaw);
		}
		sampleCounter++;
	}
}

float
updateFilter(fir_8* filt, float val)
{
	uint16_t valIndex;
	uint16_t paramIndex;
	float outval = 0.0;

	valIndex = filt->currIndex;
	filt->tabs[valIndex] = val;

	for (paramIndex = 0; paramIndex < 8; paramIndex++)
	{
		outval += (filt->params[paramIndex]) * (filt->tabs[(valIndex + paramIndex) & 0x07]);
	}

	valIndex++;
	valIndex &= 0x07;

	filt->currIndex = valIndex;

	return outval;
}

void
initFilter(fir_8* theFilter)
{
	uint8_t i;

	theFilter->currIndex = 0;

	for (i = 0; i < 8; i++)
		theFilter->tabs[i] = 0.0;

	theFilter->params[0] = 0.01;
	theFilter->params[1] = 0.05;
	theFilter->params[2] = 0.12;
	theFilter->params[3] = 0.32;
	theFilter->params[4] = 0.32;
	theFilter->params[5] = 0.12;
	theFilter->params[6] = 0.05;
	theFilter->params[7] = 0.01;
}
//******************************************************************************
uint8_t
deadline(void)
{
	uint8_t temp = 0;
	for (index = 0; index < brewing_count; index++) {
		if (time_count % brewing_list[index].deadline == 0 && brewing_list[index].running != 0 && brewing_list[index].state != finish) {
			temp = index;
			deadline_flag = 1;
			break;
		}
	}

	for (index = 0; index < brewing_count; index++) {
		if (time_count % brewing_list[index].deadline == 0 && brewing_list[index].running != 0 && brewing_list[index].state != finish && brewing_list[index].priority > brewing_list[temp].priority) {
			temp = index;
			input->state = wait;
			break;
		}
	}

	return temp;
}

uint8_t
edf(coffee_setting *pvParameters)
{
	uint8_t index;

	for (index = 0;; index++) {
		if (index * pvParameters->deadline - time_count >= 0) {
			break;
		}
	}
	return index * pvParameters->deadline - time_count;
}

uint8_t
lls(coffee_setting *pvParameters)
{
	uint8_t index;
	for (index = 0;; index++) {
		if (index * pvParameters->deadline - (pvParameters->period - pvParameters->running) >= 0) {
			break;
		}
	}
	return index * pvParameters->deadline - (pvParameters->period - pvParameters->running);
}

void
sUsedFlag(void)
{
	uint8_t index;
	uint8_t temp = 0;
	for (index = 0; index < brewing_count; index++) {
		if (!brewing_list[index].used) {
			temp = 1;
			break;
		}
	}

	if (!temp) {
		for (index = 0; index < brewing_count; index++) {
			brewing_list[index].used = 0;
		}
	}
}
//******************************************************************************
void vFPS(void *pvParameters)
{
	uint8_t index;
	uint8_t deadline_index = 0;
	uint8_t deadline_flag = 0;
	uint8_t next_index = 0;
	coffee_setting *input;

	input = pvParameters;

	for (;;)
	{
		sUsedFlag();
//-------------------------------------------------------------------------------------------
		// first task to determine which task will be excuted firstly
		if (START_FLAG) {
			if (SAME_TIME_FLAG) {
				if (input->list_index == 0 && input->running == 0 && time_count == 0) {
					if (xSemaphoreTake(xStartSemaphore, portMAX_DELAY)) {
						for (index = 0; index < brewing_count; index++) {
							if (brewing_list[index].priority > brewing_list[next_index].priority) {
								next_index = index;
							}
						}
						brewing_list[next_index].state = run;
					}
				}
			} else {
				if (input->list_index == 0 && input->running == 0 && time_count == 0) {
					input->state = run;
				}
			}
			START_FLAG = 0;
		}
//-------------------------------------------------------------------------------------------
		if (input->state == run) {
			input->running += 1;
			STM_EVAL_LEDOn(input->led_index);
			vTaskDelay(500 / portTICK_RATE_MS);
			STM_EVAL_LEDOff(input->led_index);
			vTaskDelay(500 / portTICK_RATE_MS);

			input->state = ready;
		} else if (input->state == wait) {
			for (;;) {
				if (xSemaphoreTake(input->xCoffeeSemaphore, portMAX_DELAY)) {
					input->state = run;
					break;
				}
			}
		} else if (input->state == ready) {
			deadline_flag = 0;
			deadline_index = deadline();

			if (deadline_flag) {
				input->state = wait;
				xSemaphoreGive(brewing_list[deadline_index].xCoffeeSemaphore);
			} else {
				if (input->running % input->brewing_dutation == 0 && time_count % input->period != 0) {
					input->state = finish;
				}

				for (index = 0; index < brewing_count; index++) {
					if (brewing_list[index].state != finish) {
						if (brewing_list[index].priority >= brewing_list[next_index].priority) {
							next_index = index;
						}
					}
				}


				if (next_index == input->list_index) {
					input->state = run;
				} else {
					xSemaphoreGive(brewing_list[next_index].xCoffeeSemaphore);
				}
			}
		} else if (input->state == finish) {
			xSemaphoreGive(xSoundSemaphore);
			for(;;){
				if(time_count == input->period){
					input->state = ready;
				}
			}
		}
	}
}

//******************************************************************************
void vEDF(void *pvParameters)
{
	uint8_t index;
	uint8_t next_index = 0;
	coffee_setting* input;

	input = pvParameters;

	for (;;)
	{
//-------------------------------------------------------------------------------------------
		// first task to determine which task will be excuted firstly
		if (START_FLAG) {
			if (SAME_TIME_FLAG) {
				if (input->list_index == 0 && input->running == 0 && time_count == 0) {
					if (xSemaphoreTake(xStartSemaphore, portMAX_DELAY)) {
						for (index = 0; index < brewing_count; index++) {
							if (brewing_list[index].deadline < brewing_list[next_index].deadline) {
								next_index = index;
							}
						}
						brewing_list[next_index].state = run;
					}
				}
			} else {
				if (input->list_index == 0 && input->running == 0 && time_count == 0) {
					input->state = run;
				}
			}
			START_FLAG = 0;
		}
//-------------------------------------------------------------------------------------------
		if (input->state == run) {
			time_count++;
			input->running += 1;
			STM_EVAL_LEDOn(input->led_index);
			vTaskDelay(500 / portTICK_RATE_MS);
			STM_EVAL_LEDOff(input->led_index);
			vTaskDelay(500 / portTICK_RATE_MS);

			if (input->running == input->period) {
				input->state = finish;
			} else {
				input->state = ready;
			}
		} else if (input->state == wait) {
			for (;;) {
				if (xSemaphoreTake(input->xCoffeeSemaphore, portMAX_DELAY)) {
					input->state = run;
					break;
				}
			}
		} else if (input->state == ready) {
			if (task_count > 1) {
				if (input->running % input->brewing_dutation == 0) {
					for (index = 0; index < brewing_count; index++) {
						if (brewing_list[index].state == wait) {
							next_index = index;
							break;
						}
					}

					for (index = 0; index < brewing_count; index++) {
						if (brewing_list[index].state == wait) {
							if (edf(&brewing_list[index]) < edf(&brewing_list[next_index])) {
								next_index = index;
							}
						}
					}

					input->state = wait;
					xSemaphoreGive(brewing_list[next_index].xCoffeeSemaphore);

				} else {
					input->state = run;
				}
			} else {
				input->state = run;
			}
		} else if (input->state == finish) {
			xSemaphoreGive(xSoundSemaphore);
			if (task_count > 1) {
				for (index = 0; index < brewing_count; index++) {
					if (brewing_list[index].state == wait) {
						next_index = index;
						break;
					}
				}

				for (index = 0; index < brewing_count; index++) {
					if (brewing_list[index].state == wait) {
						if (deadline(&brewing_list[index]) < deadline(&brewing_list[next_index])) {
							next_index = index;
						}
					}
				}
				xSemaphoreGive(brewing_list[next_index].xCoffeeSemaphore);
			}
			task_count--;
			vTaskDelete(xHandleBrewing[input->list_index]);
		}
	}
}
//******************************************************************************
void vLLS(void *pvParameters)
{	uint8_t index;
	uint8_t next_index = 0;
	coffee_setting* input;

	input = pvParameters;

	for (;;)
	{
//-------------------------------------------------------------------------------------------
		// first task to determine which task will be excuted firstly
		if (START_FLAG) {
			if (SAME_TIME_FLAG) {
				if (input->list_index == 0 && input->running == 0 && time_count == 0) {
					if (xSemaphoreTake(xStartSemaphore, portMAX_DELAY)) {
						for (index = 0; index < brewing_count; index++) {
							if (brewing_list[index].deadline < brewing_list[next_index].deadline) {
								next_index = index;
							}
						}
						brewing_list[next_index].state = run;
					}
				}
			} else {
				if (input->list_index == 0 && input->running == 0 && time_count == 0) {
					input->state = run;
				}
			}
			START_FLAG = 0;
		}
//-------------------------------------------------------------------------------------------
		if (input->state == run) {
			time_count++;
			input->running += 1;
			STM_EVAL_LEDOn(input->led_index);
			vTaskDelay(500 / portTICK_RATE_MS);
			STM_EVAL_LEDOff(input->led_index);
			vTaskDelay(500 / portTICK_RATE_MS);

			if (input->running == input->period) {
				input->state = finish;
			} else {
				input->state = ready;
			}
		} else if (input->state == wait) {
			for (;;) {
				if (xSemaphoreTake(input->xCoffeeSemaphore, portMAX_DELAY)) {
					input->state = run;
					break;
				}
			}
		} else if (input->state == ready) {
			if (task_count > 1) {
				if (input->running % input->brewing_dutation == 0) {
					for (index = 0; index < brewing_count; index++) {
						if (brewing_list[index].state == wait) {
							next_index = index;
							break;
						}
					}

					for (index = 0; index < brewing_count; index++) {
						if (brewing_list[index].state == wait) {
							if (lls(&brewing_list[index]) < lls(&brewing_list[next_index])) {
								next_index = index;
							}
						}
					}

					input->state = wait;
					xSemaphoreGive(brewing_list[next_index].xCoffeeSemaphore);

				} else {
					input->state = run;
				}
			} else {
				input->state = run;
			}
		} else if (input->state == finish) {
			xSemaphoreGive(xSoundSemaphore);
			if (task_count > 1) {
				for (index = 0; index < brewing_count; index++) {
					if (brewing_list[index].state == wait) {
						next_index = index;
						break;
					}
				}

				for (index = 0; index < brewing_count; index++) {
					if (brewing_list[index].state == wait) {
						if (lls(&brewing_list[index]) < lls(&brewing_list[next_index])) {
							next_index = index;
						}
					}
				}
				xSemaphoreGive(brewing_list[next_index].xCoffeeSemaphore);
			}
			task_count--;
			vTaskDelete(xHandleBrewing[input->list_index]);
		}
	}
}
