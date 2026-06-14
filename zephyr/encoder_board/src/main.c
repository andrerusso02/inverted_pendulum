#include <zephyr/kernel.h>

#include <zephyr/device.h>
#include <zephyr/drivers/sensor.h>
#include <zephyr/logging/log.h>

LOG_MODULE_REGISTER(main, LOG_LEVEL_INF);


#define USBD_NODE DT_NODELABEL(usbd)
#if DT_NODE_HAS_STATUS(USBD_NODE, okay)

#include <zephyr/usb/usbd.h>
#include <zephyr/usb/usb_ch9.h> /* Contains USB_SCD_ macros */
/* 1. Define the Device Context */

USBD_DEVICE_DEFINE(my_usbd, 
                   DEVICE_DT_GET(USBD_NODE), 
                   0x0483, 0x0001);
/* 2. Define Basic String Descriptors */
USBD_DESC_LANG_DEFINE(my_lang);
USBD_DESC_MANUFACTURER_DEFINE(my_mfr, "Zephyr");
USBD_DESC_PRODUCT_DEFINE(my_prdc, "DD's NCM Network Interface");
/* 3. Define the Full-Speed Configuration */
USBD_DESC_CONFIG_DEFINE(my_cfg_desc, "Default Config");
/* Fixed macro: removed '_ATTRIBUTES_' from the constant name */
USBD_CONFIGURATION_DEFINE(my_fs_cfg, 
                          USB_SCD_SELF_POWERED, 
                          100, 
                          &my_cfg_desc);

int start_usb_networking(void)
{
    int err;
    struct usbd_context *ctx = &my_usbd;
    /* Add Mandatory String Descriptors */
    usbd_add_descriptor(ctx, &my_lang);
    usbd_add_descriptor(ctx, &my_mfr);
    usbd_add_descriptor(ctx, &my_prdc);
    /* Setup Device Triple for IAD (Required for CDC NCM/Compound devices) */
    /* 0xEF = Miscellaneous, 0x02 = Common Class, 0x01 = Interface Association */
    usbd_device_set_code_triple(ctx, USBD_SPEED_FS, 0xEF, 0x02, 0x01);
    /* Bind Configuration and Register NCM Class */
    err = usbd_add_configuration(ctx, USBD_SPEED_FS, &my_fs_cfg);
    if (err) return err;
    /* Register all classes enabled in prj.conf (e.g. CDC NCM) */
    err = usbd_register_all_classes(ctx, USBD_SPEED_FS, 1, NULL);
    if (err) return err;    /* Initialize and Fire it up */

    /* Initialize and Fire it up */
    err = usbd_init(ctx);
    if (err) return err;

    return usbd_enable(ctx);
}
#endif


void main(void)
{
    LOG_INF("Hello from Zephyr application!");
    int err = start_usb_networking();
    if (err) {
        LOG_ERR("USB Init Failed with error: %d", err);
    }
    else {
        LOG_INF("USB Init Succeeded.");
    }

    /* Get the device binding using the compatible string */
    const struct device *const as5600_dev = DEVICE_DT_GET_ANY(ams_as5600);
    struct sensor_value rotation;
    int rc;

    if (as5600_dev == NULL) {
        LOG_ERR("No device found with compatible 'ams,as5600'");
        return;
    }

    if (!device_is_ready(as5600_dev)) {
        LOG_ERR("Device %s is not ready", as5600_dev->name);
        return;
    }

    LOG_INF("Successfully initialized AS5600 device: %s", as5600_dev->name);

    while (1) {
        /* Fetch the latest sample from the sensor */
        rc = sensor_sample_fetch(as5600_dev);
        
        if (rc != 0) {
            LOG_ERR("Sensor fetch failed or magnet issue: %d", rc);
        } else {
            /* Retrieve the rotation data */
            rc = sensor_channel_get(as5600_dev, SENSOR_CHAN_ROTATION, &rotation);
            if (rc == 0) {
                /* val1 contains the integer part (degrees), val2 contains the fractional part (micro-degrees) */
                LOG_INF("Rotation: %d.%06d degrees", rotation.val1, rotation.val2);
            } else {
                LOG_ERR("Failed to get sensor channel data: %d", rc);
            }
        }

        /* Wait 10 millisecondss before reading again */
        k_msleep(10);
    }
}
