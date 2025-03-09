#ifndef CUDAGRAVSUM_H
#define CUDAGRAVSUM_H

#include "treedefs.h"

void cuda_gravsum(bodyptr current_body, cell_ll_entry_t *cell_list_tail, cell_ll_entry_t *body_list_tail);

void cuda_gravsum_init();

void cuda_gravsum_dispatch();

void cuda_update_body_cell_data();

void cuda_dispatch_last_batch();

void cuda_collect_result();

void cuda_free_all();
#endif //CUDAGRAVSUM_H
