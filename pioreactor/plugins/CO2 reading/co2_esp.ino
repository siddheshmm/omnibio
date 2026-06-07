#include <WiFi.h>
#include <PubSubClient.h>
#include <SensirionI2cScd4x.h>
#include <Wire.h>

// WiFi credentials
const char *WIFI_SSID = "sid";
const char *WIFI_PASSWORD = "hellosidd";

// MQTT broker
const char *MQTT_BROKER = "10.102.100.34";
const int MQTT_PORT = 1883;
const char *MQTT_USER = "pioreactor";
const char *MQTT_PASSWORD = "raspberry";
const char *UNIT = "pioreactor01";

// Topics
char topic_co2[100];
char topic_temp[100];
char topic_hum[100];

WiFiClient espClient;
PubSubClient mqtt(espClient);
SensirionI2cScd4x scd4x;

void setup_wifi()
{
    WiFi.begin(WIFI_SSID, WIFI_PASSWORD);
    Serial.print("Connecting to WiFi");
    while (WiFi.status() != WL_CONNECTED)
    {
        delay(500);
        Serial.print(".");
    }
    Serial.println("\nWiFi connected: " + WiFi.localIP().toString());
}

void reconnect_mqtt()
{
    while (!mqtt.connected())
    {
        Serial.print("Connecting to MQTT...");
        if (mqtt.connect("esp32_scd4x", MQTT_USER, MQTT_PASSWORD))
        {
            Serial.println("connected");
            // Subscribe to get current experiment name
            // mqtt.subscribe("pioreactor/+/$experiment");
            char experiment_topic[100];

            snprintf(experiment_topic, 100,
                     "pioreactor/%s/$experiment", UNIT);

            mqtt.subscribe(experiment_topic);
        }
        else
        {
            Serial.print("failed, rc=");
            Serial.println(mqtt.state());
            delay(5000);
        }
    }
}

// Track current experiment
char current_experiment[64] = "$experiment";

void mqtt_callback(char *topic, byte *payload, unsigned int length)
{
    // Update experiment name if received
    char msg[64];
    memcpy(msg, payload, min(length, (unsigned int)63));
    msg[length] = '\0';
    strncpy(current_experiment, msg, 63);
    Serial.print("Experiment: ");
    Serial.println(current_experiment);
}

void setup()
{
    Serial.begin(115200);
    Wire.begin();

    // Init SCD4x
    scd4x.begin(Wire, SCD41_I2C_ADDR_62);
    scd4x.stopPeriodicMeasurement();
    delay(500);
    scd4x.startPeriodicMeasurement();
    Serial.println("SCD4x started");

    setup_wifi();
    mqtt.setServer(MQTT_BROKER, MQTT_PORT);
    mqtt.setCallback(mqtt_callback);
}

void loop()
{
    if (!mqtt.connected())
        reconnect_mqtt();
    mqtt.loop();

    // Read every 30 seconds
    static unsigned long last_read = 0;
    if (millis() - last_read > 30000)
    {
        last_read = millis();

        bool dataReady = false;
        scd4x.getDataReadyStatus(dataReady);

        if (dataReady)
        {
            uint16_t co2 = 0;
            float temp = 0, hum = 0;
            int16_t err = scd4x.readMeasurement(co2, temp, hum);

            if (err == 0)
            {
                // Build topics
                snprintf(topic_co2, 100, "pioreactor/%s/%s/co2_reading/co2", UNIT, current_experiment);
                snprintf(topic_temp, 100, "pioreactor/%s/%s/co2_reading/temperature", UNIT, current_experiment);
                snprintf(topic_hum, 100, "pioreactor/%s/%s/co2_reading/relative_humidity", UNIT, current_experiment);

                // Publish
                char buf[20];
                snprintf(buf, 20, "%u", co2);
                mqtt.publish(topic_co2, buf);

                snprintf(buf, 20, "%.2f", temp);
                mqtt.publish(topic_temp, buf);

                snprintf(buf, 20, "%.2f", hum);
                mqtt.publish(topic_hum, buf);

                Serial.printf("Published — CO2: %uppm, Temp: %.1fC, Hum: %.1f%%\n", co2, temp, hum);
                // Serial.print("Published - CO2: ");
                // Serial.print(co2);
                // Serial.print("ppm, Temp: ");
                // Serial.print(temp, 1);
                // Serial.print("C, Hum: ");
                // Serial.print(hum, 1);
                // Serial.println("%");
            }
        }
    }
}