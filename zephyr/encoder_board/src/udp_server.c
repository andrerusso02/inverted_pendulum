#include "udp_server.h"
#include <zephyr/net/socket.h>
#include <arpa/inet.h>

static int sock = -1;
static struct sockaddr_in destination_addr;

int udp_server_init(void)
{
    sock = socket(AF_INET, SOCK_DGRAM, IPPROTO_UDP);
    if (sock < 0) {
        return -1;
    }

    destination_addr.sin_family = AF_INET;
    destination_addr.sin_port = htons(SERVER_PORT);
    inet_pton(AF_INET, DEST_ADDR, &destination_addr.sin_addr);

    return 0;
}

int udp_server_send(const void *data, size_t len)
{
    if (sock < 0) {
        return -1;
    }

    return sendto(sock, data, len, 0, 
                  (struct sockaddr *)&destination_addr, 
                  sizeof(destination_addr));
}