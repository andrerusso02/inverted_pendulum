#ifndef UDP_SERVER_H
#define UDP_SERVER_H

#include <stddef.h>

#define DEST_ADDR "255.255.255.255"
#define SERVER_PORT 5005

int udp_server_init(void);

int udp_server_send(const void *data, size_t len);

#endif /* UDP_SERVER_H */