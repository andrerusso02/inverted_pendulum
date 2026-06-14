#include <zephyr/kernel.h>


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
    printk("Hello from Zephyr application!\n");
    int err = start_usb_networking();
    if (err) {
        printk("USB Init Failed with error: %d\n", err);
    }
    else {
        printk("USB Init Succeeded.\n");
    }
}