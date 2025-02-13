import torch
import torch.nn.functional as F
from typing import Iterator
import pandas as pd
import numpy as np
import time
import asyncio
from typing import List
import copy
import random

from AgentDropout.graph.graph import Graph
from experiments.accuracy import Accuracy
from AgentDropout.utils.globals import Cost, PromptTokens, CompletionTokens
from AgentDropout.utils.utils import nuclear_norm,frobenius_norm

async def train(graph:Graph,
            dataset,
            num_iters:int=100,
            num_rounds:int=1,
            lr:float=0.1,
            batch_size:int = 4,
            imp_per_iters: int = 1,
            pruning_rate: float = 0.05,
            args=None,
            kwargs=None,
          ) -> None:
    
    def infinite_data_loader() -> Iterator[pd.DataFrame]:
        while True:
            for idx in range(len(dataset)):
                record = dataset[idx]
                yield record
    
    loader = infinite_data_loader()
    if args.dec:
        graph.optimized_spatial=False
        graph.optimized_temporal=False
        
        if not graph.diff:
            optimizer = torch.optim.Adam([graph.spatial_logits_1,graph.temporal_logits_1], lr=lr)
        # elif graph.diff and graph.dec_1:
        #     optimizer = torch.optim.Adam(list(graph.spatial_logits.parameters()) + list(graph.temporal_logits.parameters())+[graph.decision_logits],lr=lr)
        else:
            optimizer = torch.optim.Adam(list(graph.spatial_logits_1.parameters()) + list(graph.temporal_logits_1.parameters()),lr=lr)
        for i_iter in range(2):
            print(f"Train {i_iter}", 80*'-')
            start_ts = time.time()
            correct_answers = []
            answer_log_probs = []
            add_losses = []
            for i_record, record in zip(range(20), loader):
                realized_graph = copy.deepcopy(graph)
                realized_graph.spatial_logits_1 = graph.spatial_logits_1
                realized_graph.temporal_logits_1 = graph.temporal_logits_1
                # if graph.dec_1:
                #     realized_graph.decision_logits = graph.decision_logits
                
                # print(graph.spatial_logits)
                if not graph.diff:
                    spatial_matrix_train = realized_graph.spatial_logits_1.reshape((sum(args.agent_nums),sum(args.agent_nums)))
                    temporal_matrix_train = realized_graph.temporal_logits_1.reshape((sum(args.agent_nums),sum(args.agent_nums)))
                else:
                    spatial_matrix_train = [param.reshape((sum(args.agent_nums), sum(args.agent_nums))) for param in realized_graph.spatial_logits_1]
                    temporal_matrix_train = [param.reshape((sum(args.agent_nums), sum(args.agent_nums))) for param in realized_graph.temporal_logits_1]

                spatial_matrix_fixed = torch.tensor(kwargs["fixed_spatial_masks"],dtype=torch.float32).reshape((sum(args.agent_nums),sum(args.agent_nums)))
                temporal_matrix_fixed = torch.tensor(kwargs["fixed_temporal_masks"],dtype=torch.float32).reshape((sum(args.agent_nums),sum(args.agent_nums)))
                if not graph.diff:
                    loss_s = nuclear_norm(spatial_matrix_train)
                    loss_t = nuclear_norm(temporal_matrix_train)
                    frob_loss_s = frobenius_norm(spatial_matrix_fixed, spatial_matrix_train)
                    frob_loss_t = frobenius_norm(temporal_matrix_fixed, temporal_matrix_train)
                    loss_cs = connectivity_loss(spatial_matrix_train)
                    loss_ct = connectivity_loss(temporal_matrix_train)
                    # print(loss_cs)
                else:
                    # loss_s = sum(nuclear_norm(matrix) for matrix in spatial_matrix_train)
                    # loss_t = sum(nuclear_norm(matrix) for matrix in temporal_matrix_train)
                    # frob_loss_s = sum(frobenius_norm(spatial_matrix_fixed, matrix) for matrix in spatial_matrix_train)
                    # frob_loss_t = sum(frobenius_norm(temporal_matrix_fixed, matrix) for matrix in temporal_matrix_train)
                    loss_s = torch.mean(torch.stack([nuclear_norm(matrix) for matrix in spatial_matrix_train]))
                    loss_t = torch.mean(torch.stack([nuclear_norm(matrix) for matrix in temporal_matrix_train]))
                    loss_cs = torch.mean(torch.stack([connectivity_loss(matrix) for matrix in spatial_matrix_train]))
                    loss_ct = torch.mean(torch.stack([connectivity_loss(matrix) for matrix in temporal_matrix_train]))
                    frob_loss_s = torch.mean(torch.stack([frobenius_norm(spatial_matrix_fixed, matrix) for matrix in spatial_matrix_train]))
                    frob_loss_t = torch.mean(torch.stack([frobenius_norm(temporal_matrix_fixed, matrix) for matrix in temporal_matrix_train]))
                add_loss = loss_s + loss_t + F.relu(frob_loss_s - args.delta) + F.relu(frob_loss_t - args.delta)
                add_loss = 0
                input_dict = dataset.record_to_input(record)
                print(input_dict)
                if args.dec:
                    answer_log_probs.append(asyncio.create_task(realized_graph.arun(input_dict,num_rounds,skip=True)))
                else:
                    answer_log_probs.append(asyncio.create_task(realized_graph.arun(input_dict,num_rounds)))
                correct_answer = dataset.record_to_target_answer(record)
                correct_answers.append(correct_answer)
                add_losses.append(add_loss)
                
            raw_results = await asyncio.gather(*answer_log_probs)
            raw_answers, log_probs = zip(*raw_results)
            loss_list: List[torch.Tensor] = []
            utilities: List[float] = []
            answers: List[str] = []
            
            for raw_answer, log_prob, add_loss, correct_answer in zip(raw_answers, log_probs, add_losses, correct_answers):
                answer = dataset.postprocess_answer(raw_answer)
                answers.append(answer)
                assert isinstance(correct_answer, str), \
                        f"String expected but got {correct_answer} of type {type(correct_answer)} (1)"
                accuracy = Accuracy()
                accuracy.update(answer, correct_answer)
                utility = accuracy.get()
                utilities.append(utility)
                single_loss = - log_prob * utility
                loss_list.append(single_loss+add_loss)
                print(f"correct answer:{correct_answer}")
        
            total_loss = torch.mean(torch.stack(loss_list))
            optimizer.zero_grad() 
            total_loss.backward()
            optimizer.step()
            if not graph.diff:
                spatial_probs = torch.sigmoid(graph.spatial_logits_1)
                temporal_probs = torch.sigmoid(graph.temporal_logits_1)
            else:
                spatial_probs = [torch.sigmoid(logit) for logit in graph.spatial_logits_1]
                temporal_probs = [torch.sigmoid(logit) for logit in graph.temporal_logits_1]
            
            print("raw_answers:",raw_answers)
            print("answers:",answers)
            print(f"Batch time {time.time() - start_ts:.3f}")
            print("utilities:", utilities)
            print("loss:", total_loss.item())
            print("Spatial logits Grad:", graph.spatial_logits_1[0].grad)
            # print("Temporal logits Grad:", graph.spatial_logits.grad)
            print("Spatial logits:", graph.spatial_logits_1)
            print("Temporal logits:", graph.temporal_logits_1)
            # if graph.dec_1:
            #     print("Decision logits:", graph.decision_logits)
            print("Spatial probs:", spatial_probs)
            print("Temporal probs:", temporal_probs)
            print(f"Cost {Cost.instance().value}")
            print(f"PromptTokens {PromptTokens.instance().value}")
            print(f"CompletionTokens {CompletionTokens.instance().value}")
        graph.update_masks_dec()

    loader = infinite_data_loader()

    if not graph.diff:
        optimizer = torch.optim.Adam([graph.spatial_logits,graph.temporal_logits], lr=lr)
    # elif graph.diff and graph.dec_1:
    #     optimizer = torch.optim.Adam(list(graph.spatial_logits.parameters()) + list(graph.temporal_logits.parameters())+[graph.decision_logits],lr=lr)
    else:
        optimizer = torch.optim.Adam(list(graph.spatial_logits.parameters()) + list(graph.temporal_logits.parameters()),lr=lr)
    
    graph.optimized_spatial=True
    graph.optimized_temporal=True

    for i_iter in range(2):
        print(f"Train {i_iter}", 80*'-')
        start_ts = time.time()
        correct_answers = []
        answer_log_probs = []
        add_losses = []
        for i_record, record in zip(range(10), loader):
            realized_graph = copy.deepcopy(graph)
            realized_graph.spatial_logits = graph.spatial_logits
            realized_graph.temporal_logits = graph.temporal_logits
            # if graph.dec_1:
            #     realized_graph.decision_logits = graph.decision_logits
            
            # print(graph.spatial_logits)
            if not graph.diff:
                spatial_matrix_train = realized_graph.spatial_logits.reshape((sum(args.agent_nums),sum(args.agent_nums)))
                temporal_matrix_train = realized_graph.temporal_logits.reshape((sum(args.agent_nums),sum(args.agent_nums)))
                # spatial_matrix_train = remove_zero_rows_and_columns(spatial_matrix_train)
                # temporal_matrix_train = remove_zero_rows_and_columns(temporal_matrix_train)
            else:
                spatial_matrix_train = [param.reshape((sum(args.agent_nums), sum(args.agent_nums))) for param in realized_graph.spatial_logits]
                temporal_matrix_train = [param.reshape((sum(args.agent_nums), sum(args.agent_nums))) for param in realized_graph.temporal_logits]
                # spatial_matrix_train = [remove_zero_rows_and_columns(param,i,i) for param,i in zip(spatial_matrix_train,graph.skip_nodes)]
                # # temporal_matrix_train = [remove_zero_rows_and_columns(param) for param in temporal_matrix_train]
                # for i in range(len(temporal_matrix_train)):
                #     temporal_matrix_train[i] = remove_zero_rows_and_columns(temporal_matrix_train[i],graph.skip_nodes[i],graph.skip_nodes[i+1])

            spatial_matrix_fixed = torch.tensor(kwargs["fixed_spatial_masks"],dtype=torch.float32).reshape((sum(args.agent_nums),sum(args.agent_nums)))
            temporal_matrix_fixed = torch.tensor(kwargs["fixed_temporal_masks"],dtype=torch.float32).reshape((sum(args.agent_nums),sum(args.agent_nums)))
            # spatial_matrix_fixed = spatial_matrix_fixed[:4,:4]
            # temporal_matrix_fixed = temporal_matrix_fixed[:4,:4]
            if not graph.diff:
                loss_s = nuclear_norm(spatial_matrix_train)
                loss_t = nuclear_norm(temporal_matrix_train)
                frob_loss_s = frobenius_norm(spatial_matrix_fixed, spatial_matrix_train)
                frob_loss_t = frobenius_norm(temporal_matrix_fixed, temporal_matrix_train)
                loss_cs = connectivity_loss(spatial_matrix_train)
                loss_ct = connectivity_loss(temporal_matrix_train)
                # print(loss_cs)
            else:
                # loss_s = sum(nuclear_norm(matrix) for matrix in spatial_matrix_train)
                # loss_t = sum(nuclear_norm(matrix) for matrix in temporal_matrix_train)
                # frob_loss_s = sum(frobenius_norm(spatial_matrix_fixed, matrix) for matrix in spatial_matrix_train)
                # frob_loss_t = sum(frobenius_norm(temporal_matrix_fixed, matrix) for matrix in temporal_matrix_train)
                loss_s = torch.mean(torch.stack([nuclear_norm(matrix) for matrix in spatial_matrix_train]))
                loss_t = torch.mean(torch.stack([nuclear_norm(matrix) for matrix in temporal_matrix_train]))
                loss_cs = torch.mean(torch.stack([connectivity_loss(matrix) for matrix in spatial_matrix_train]))
                loss_ct = torch.mean(torch.stack([connectivity_loss(matrix) for matrix in temporal_matrix_train]))
                frob_loss_s = torch.mean(torch.stack([frobenius_norm(spatial_matrix_fixed, matrix) for matrix in spatial_matrix_train]))
                frob_loss_t = torch.mean(torch.stack([frobenius_norm(temporal_matrix_fixed, matrix) for matrix in temporal_matrix_train]))
            add_loss = loss_s + loss_t + F.relu(frob_loss_s - args.delta) + F.relu(frob_loss_t - args.delta)
            # if graph.diff:
            # add_loss = 0
            input_dict = dataset.record_to_input(record)
            print(input_dict)
            if args.dec:
                answer_log_probs.append(asyncio.create_task(realized_graph.arun(input_dict,num_rounds)))
            else:
                answer_log_probs.append(asyncio.create_task(realized_graph.arun(input_dict,num_rounds)))
            correct_answer = dataset.record_to_target_answer(record)
            correct_answers.append(correct_answer)
            add_losses.append(add_loss)
            
        raw_results = await asyncio.gather(*answer_log_probs)
        raw_answers, log_probs = zip(*raw_results)
        loss_list: List[torch.Tensor] = []
        utilities: List[float] = []
        answers: List[str] = []
        
        for raw_answer, log_prob, add_loss, correct_answer in zip(raw_answers, log_probs, add_losses, correct_answers):
            answer = dataset.postprocess_answer(raw_answer)
            answers.append(answer)
            assert isinstance(correct_answer, str), \
                    f"String expected but got {correct_answer} of type {type(correct_answer)} (1)"
            accuracy = Accuracy()
            accuracy.update(answer, correct_answer)
            utility = accuracy.get()
            utilities.append(utility)
            single_loss = - log_prob * utility
            loss_list.append(single_loss+add_loss)
            print(f"correct answer:{correct_answer}")
    
        total_loss = torch.mean(torch.stack(loss_list))
        optimizer.zero_grad() 
        total_loss.backward()
        optimizer.step()
        if not graph.diff:
            spatial_probs = torch.sigmoid(graph.spatial_logits)
            temporal_probs = torch.sigmoid(graph.temporal_logits)
        else:
            spatial_probs = [torch.sigmoid(logit) for logit in graph.spatial_logits]
            temporal_probs = [torch.sigmoid(logit) for logit in graph.temporal_logits]
        
        print("raw_answers:",raw_answers)
        print("answers:",answers)
        print(f"Batch time {time.time() - start_ts:.3f}")
        print("utilities:", utilities)
        print("loss:", total_loss.item()) 
        # print("Spatial logits Grad:", graph.spatial_logits.grad)
        # print("Temporal logits Grad:", graph.spatial_logits.grad)
        print("Spatial logits:", graph.spatial_logits)
        print("Temporal logits:", graph.temporal_logits)
        # if graph.dec_1:
        #     print("Decision logits:", graph.decision_logits)
        print("Spatial probs:", spatial_probs)
        print("Temporal probs:", temporal_probs)
        print(f"Cost {Cost.instance().value}")
        print(f"PromptTokens {PromptTokens.instance().value}")
        print(f"CompletionTokens {CompletionTokens.instance().value}")
        
        if (i_iter+1)%2 == 0:
            if not graph.diff:
                spatial_masks, temporal_masks = graph.update_masks(pruning_rate)
            else:
                spatial_masks, temporal_masks = graph.update_masks_diff(pruning_rate)
            print("spatial masks:",spatial_masks)
            print("temporal masks:",temporal_masks)
            if not graph.diff:
                print("spatial sparsity:",spatial_masks.sum()/spatial_masks.numel())
                print("temporal sparsity:",temporal_masks.sum()/temporal_masks.numel())
            else:
                print("spatial sparsity:",spatial_masks[0].sum()/spatial_masks[0].numel())
                print("temporal sparsity:",temporal_masks[0].sum()/temporal_masks[0].numel())

    # graph.optimized_spatial=False
    # graph.optimized_temporal=False        


def connectivity_loss(A: torch.Tensor) -> torch.Tensor:

    expA = torch.matrix_exp(A)

    penalty = torch.trace(expA) - A.shape[0]
    return penalty

def set_seed(seed):
    torch.manual_seed(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    random.seed(seed)
    np.random.seed(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def remove_zero_rows_and_columns(matrix,row_to_delete,col_to_delete):
    new_matrix = torch.cat([matrix[:row_to_delete], matrix[row_to_delete+1:]])
    new_matrix = torch.cat([new_matrix[:, :col_to_delete], new_matrix[:, col_to_delete+1:]], dim=1)

    return new_matrix