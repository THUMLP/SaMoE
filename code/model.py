from numpy.core.fromnumeric import size
from torch import dropout, int_repr, nn
from transformers import RobertaModel, RobertaConfig

from transformers.models.roberta.modeling_roberta import RobertaEncoder, RobertaForSequenceClassification, RobertaLayer, BaseModelOutputWithPastAndCrossAttentions, RobertaPreTrainedModel, RobertaEmbeddings
from torch.utils.checkpoint import checkpoint
import torch
import torch.nn.functional as F
from torch.distributions.normal import Normal
from typing_extensions import final
import numpy as np

class RobertaClassificationHead_without_log(nn.Module):
    """Head for sentence-level classification tasks without log operation."""

    def __init__(self, hidden_size, hidden_dropout_prob, num_labels):
        super().__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(hidden_dropout_prob)
        self.out_proj = nn.Linear(hidden_size, num_labels)
    def forward(self, x):
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x

class RobertaMoEForSequenceClassification(nn.Module):
    def __init__(self, config,num_public_layers=12, num_experts=8 ,num_labels=2, num_gate_layer=2):
        super(RobertaMoEForSequenceClassification, self).__init__()
        config.gradient_checkpointing = True
        self.config = config
        self.num_labels = 2
        self.num_experts = num_experts
        self.roberta = RobertaMoEModel(config, num_public_layers=num_public_layers, num_experts=num_experts, num_gate_layer=num_gate_layer)
        self.classifiers = nn.ModuleList([RobertaClassificationHead_without_log(config.hidden_size, config.hidden_dropout_prob, num_labels) for _ in range(num_experts)])
        self.cross_entropy = nn.CrossEntropyLoss(reduction='none')
    
    def load_roberta(self, roberta_path):
        """
            load RoBERTa parameters into SaMoE.
        """
        roberta_dict = torch.load(roberta_path+'/pytorch_model.bin')
        model_dict = self.state_dict()
        for n in roberta_dict:
            if n.startswith('roberta'):
                if n in model_dict:
                    model_dict[n] = roberta_dict[n]
                elif n.startswith('roberta.encoder.layer.'): # parameter related to the experts encoder
                    new_n = n[22:]
                    buf = ''
                    for i in new_n:
                        if i.isdigit():
                            buf += i
                        else:
                            break
                    num = int(buf) - 12
                    for i in range(self.num_experts):
                        new_n = n.replace('roberta.encoder.layer.{}'.format(buf),'roberta.encoder.expert_layer.{}.{}'.format(i, num))
                        model_dict[new_n] = roberta_dict[n]
                else:
                    print(n)
            else:
                print(n)
        self.load_state_dict(model_dict)
        print('roberta loaded successfully.')
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`):
            Labels for computing the sequence classification/regression loss. Indices should be in :obj:`[0, ...,
            config.num_labels - 1]`. If :obj:`config.num_labels == 1` a regression loss is computed (Mean-Square loss),
            If :obj:`config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        # return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs, origin_gates = self.roberta(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states
        )

        final_out = []
        for i in range(self.num_experts):
            final_out.append(self.classifiers[i](outputs[i][:,0,:]).unsqueeze(1)) # bsz * 1 * 2
        final_out_mat = torch.nn.functional.softmax(torch.cat(final_out, dim=1), dim=2) # bsz * num of exp * 2
        gate_prob = torch.softmax(origin_gates, dim=-1)
        
        final_out_logits = torch.bmm(gate_prob.unsqueeze(1), final_out_mat).squeeze(1)
        loss = []
        for logits in final_out:
            loss.append(self.cross_entropy(logits.squeeze(1), labels.view(-1)).view(-1,1))
        if len(loss) == 1:
            loss_mat = loss[0].view(-1,1)
        else:
            loss_mat = torch.cat(loss, dim=1) # bsz * # of expert
        final_loss = torch.mean(torch.sum(gate_prob * loss_mat, dim=1))
        return final_out, final_loss, final_out_logits, origin_gates

class RobertaMoEModel(RobertaPreTrainedModel):

    _keys_to_ignore_on_load_missing = [r"position_ids"]

    # Copied from transformers.models.bert.modeling_bert.BertModel.__init__ with Bert->Roberta
    def __init__(self, config, num_public_layers=12, num_experts=8, num_gate_layer=2):
        super().__init__(config)
        self.config = config
        self.encoder = RobertaMoEEncoder(config, num_public_layers=num_public_layers, num_experts=num_experts, num_gate_layer=num_gate_layer)
        self.embeddings = RobertaEmbeddings(config)

    def get_input_embeddings(self):
        return self.embeddings.word_embeddings

    def set_input_embeddings(self, value):
        self.embeddings.word_embeddings = value

    def _prune_heads(self, heads_to_prune):
        """
        Prunes heads of the model. heads_to_prune: dict of {layer_num: list of heads to prune in this layer} See base
        class PreTrainedModel
        """
        for layer, heads in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(heads)

    # Copied from transformers.models.bert.modeling_bert.BertModel.forward
    
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        past_key_values=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None
    ):
        r"""
        encoder_hidden_states  (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length, hidden_size)`, `optional`):
            Sequence of hidden-states at the output of the last layer of the encoder. Used in the cross-attention if
            the model is configured as a decoder.
        encoder_attention_mask (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
            Mask to avoid performing attention on the padding token indices of the encoder input. This mask is used in
            the cross-attention if the model is configured as a decoder. Mask values selected in ``[0, 1]``: ``1`` for
            tokens that are NOT MASKED, ``0`` for MASKED tokens.
        past_key_values (:obj:`tuple(tuple(torch.FloatTensor))` of length :obj:`config.n_layers` with each tuple having 4 tensors of shape :obj:`(batch_size, num_heads, sequence_length - 1, embed_size_per_head)`):
            Contains precomputed key and value hidden states of the attention blocks. Can be used to speed up decoding.

            If :obj:`past_key_values` are used, the user can optionally input only the last :obj:`decoder_input_ids`
            (those that don't have their past key value states given to this model) of shape :obj:`(batch_size, 1)`
            instead of all :obj:`decoder_input_ids` of shape :obj:`(batch_size, sequence_length)`.
        use_cache (:obj:`bool`, `optional`):
            If set to :obj:`True`, :obj:`past_key_values` key value states are returned and can be used to speed up
            decoding (see :obj:`past_key_values`).
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        #return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        if not self.config.is_decoder:
            use_cache = False

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
            batch_size, seq_length = input_shape
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
            batch_size, seq_length = input_shape
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        device = input_ids.device if input_ids is not None else inputs_embeds.device

        # past_key_values_length
        past_key_values_length = past_key_values[0][0].shape[2] if past_key_values is not None else 0

        if attention_mask is None:
            attention_mask = torch.ones(((batch_size, seq_length + past_key_values_length)), device=device)
        if token_type_ids is None:
            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)

        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        extended_attention_mask: torch.Tensor = self.get_extended_attention_mask(attention_mask, input_shape, device)

        # If a 2D or 3D attention mask is provided for the cross-attention
        # we need to make broadcastable to [batch_size, num_heads, seq_length, seq_length]
        if self.config.is_decoder and encoder_hidden_states is not None:
            encoder_batch_size, encoder_sequence_length, _ = encoder_hidden_states.size()
            encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)
            if encoder_attention_mask is None:
                encoder_attention_mask = torch.ones(encoder_hidden_shape, device=device)
            encoder_extended_attention_mask = self.invert_attention_mask(encoder_attention_mask)
        else:
            encoder_extended_attention_mask = None

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
        # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)

        embedding_output = self.embeddings(
            input_ids=input_ids,
            position_ids=position_ids,
            token_type_ids=token_type_ids,
            inputs_embeds=inputs_embeds,
            past_key_values_length=past_key_values_length,
        )
        encoder_outputs, gates = self.encoder(
            embedding_output,
            attention_mask=extended_attention_mask,
            head_mask=head_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_extended_attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states
        )
        

        return encoder_outputs,gates

class RobertaMoEEncoder(nn.Module):
    def __init__(self, config, num_public_layers, num_experts, num_gate_layer=2):
        super(RobertaMoEEncoder,self).__init__()
        self.config = config
        self.layer = nn.ModuleList([RobertaLayer(config) for _ in range(num_public_layers)])
        self.expert_layer = nn.ModuleList([nn.ModuleList([RobertaLayer(config) for _ in range(config.num_hidden_layers-num_public_layers)]) for _ in range(num_experts)])
        self.num_gate_layer = num_gate_layer
        self.gate_layer = nn.ModuleList([RobertaLayer(config) for _ in range(self.num_gate_layer)])
        self.gate_classifier = RobertaClassificationHead_without_log(1024, config.hidden_dropout_prob, num_experts)
    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        past_key_values=None,
        use_cache=None,
        output_attentions=False,
        output_hidden_states=False
    ):
        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None
        all_cross_attentions = () if output_attentions and self.config.add_cross_attention else None

        next_decoder_cache = () if use_cache else None
        for i, layer_module in enumerate(self.layer):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            layer_head_mask = head_mask[i] if head_mask is not None else None
            past_key_value = past_key_values[i] if past_key_values is not None else None
            if getattr(self.config, "gradient_checkpointing", False):

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return module(*inputs, past_key_value, output_attentions)

                    return custom_forward

                layer_outputs = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(layer_module),
                    hidden_states,
                    attention_mask,
                    layer_head_mask,
                    encoder_hidden_states,
                    encoder_attention_mask,
                )
            else:
                layer_outputs = layer_module(
                    hidden_states,
                    attention_mask,
                    layer_head_mask,
                    encoder_hidden_states,
                    encoder_attention_mask,
                    past_key_value,
                    output_attentions,
                )

            hidden_states = layer_outputs[0]
            if use_cache:
                next_decoder_cache += (layer_outputs[-1],)
            if output_attentions:
                all_self_attentions = all_self_attentions + (layer_outputs[1],)
                if self.config.add_cross_attention:
                    all_cross_attentions = all_cross_attentions + (layer_outputs[2],)
        hidden_states_inter = torch.clone(hidden_states)
        for i, layer_module in enumerate(self.gate_layer):
            if getattr(self.config, "gradient_checkpointing", False):

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return module(*inputs, past_key_value, output_attentions)

                    return custom_forward

                layer_outputs = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(layer_module),
                    hidden_states,
                    attention_mask,
                    None,
                    encoder_hidden_states,
                    encoder_attention_mask,
                )
            else:
                layer_outputs = layer_module(
                        hidden_states,
                        attention_mask,
                        None,
                        encoder_hidden_states,
                        encoder_attention_mask,
                        None,
                        output_attentions,
                )
            hidden_states = layer_outputs[0]
        gates = self.gate_classifier(hidden_states[:,0,:])
        
        hidden_states_list = []
        
        for j, expert_module in enumerate(self.expert_layer):
            for i, layer_module in enumerate(expert_module):
                if i==0:
                    hidden_states = hidden_states_inter
                else:
                    hidden_states = hidden_states_list[j]
                if output_hidden_states:
                    all_hidden_states = all_hidden_states + (hidden_states,)

                layer_head_mask = head_mask[i] if head_mask is not None else None
                past_key_value = past_key_values[i] if past_key_values is not None else None
                if getattr(self.config, "gradient_checkpointing", False):

                    def create_custom_forward(module):
                        def custom_forward(*inputs):
                            return module(*inputs, past_key_value, output_attentions)

                        return custom_forward

                    layer_outputs = torch.utils.checkpoint.checkpoint(
                        create_custom_forward(layer_module),
                        hidden_states,
                        attention_mask,
                        layer_head_mask,
                        encoder_hidden_states,
                        encoder_attention_mask,
                    )
                else:
                    layer_outputs = layer_module(
                        hidden_states,
                        attention_mask,
                        layer_head_mask,
                        encoder_hidden_states,
                        encoder_attention_mask,
                        past_key_value,
                        output_attentions,
                    )

                hidden_states = layer_outputs[0]
                if i == 0:
                    hidden_states_list.append(hidden_states)
                else:
                    hidden_states_list[j] = hidden_states
                if use_cache:
                    next_decoder_cache += (layer_outputs[-1],)
                if output_attentions:
                    all_self_attentions = all_self_attentions + (layer_outputs[1],)
                    if self.config.add_cross_attention:
                        all_cross_attentions = all_cross_attentions + (layer_outputs[2],)
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        return hidden_states_list, gates

class RobertaMoEForSequenceClassification_adaptive(nn.Module):
    def __init__(self, config,num_public_layers=12, num_experts=8 ,num_labels=2, num_gate_layer=2):
        super(RobertaMoEForSequenceClassification_adaptive, self).__init__()
        config.gradient_checkpointing = True
        self.config = config
        self.num_labels = 2
        self.num_experts = num_experts
        self.roberta = RobertaMoEModel_adaptive(config, num_public_layers=num_public_layers, num_experts=num_experts, num_gate_layer=num_gate_layer)
        self.classifiers = nn.ModuleList([RobertaClassificationHead_without_log(config.hidden_size, config.hidden_dropout_prob, num_labels) for _ in range(num_experts)])
        self.cross_entropy = nn.CrossEntropyLoss(reduction='none')
    def load_roberta(self, roberta_path):
        """
            load RoBERTa parameters into SaMoE.
        """
        roberta_dict = torch.load(roberta_path+'/pytorch_model.bin')
        model_dict = self.state_dict()
        for n in roberta_dict:
            if n.startswith('roberta'):
                if n in model_dict:
                    model_dict[n] = roberta_dict[n]
                elif n.startswith('roberta.encoder.layer.'): # parameter related to the experts encoder
                    new_n = n[22:]
                    buf = ''
                    for i in new_n:
                        if i.isdigit():
                            buf += i
                        else:
                            break
                    num = int(buf) - 12
                    for i in range(self.num_experts):
                        new_n = n.replace('roberta.encoder.layer.{}'.format(buf),'roberta.encoder.expert_layer.{}.{}'.format(i, num))
                        model_dict[new_n] = roberta_dict[n]
        self.load_state_dict(model_dict)

    def load_MoE(self, MoE_path):
        """
            load parameters of MoE to SaMoE
        """
        MoE_dict = torch.load(MoE_path+'/pytorch_model.bin')
        model_dict = self.state_dict()
        for n in MoE_dict:
            if n in model_dict:
                model_dict[n] = MoE_dict[n]
            else:
                print(n)
        self.load_state_dict(model_dict)
        print('MoE loaded successfully.')

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`):
            Labels for computing the sequence classification/regression loss. Indices should be in :obj:`[0, ...,
            config.num_labels - 1]`. If :obj:`config.num_labels == 1` a regression loss is computed (Mean-Square loss),
            If :obj:`config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        #return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs, origin_gates, shifted_gates = self.roberta(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states
        )

        final_out = []
        for i in range(self.num_experts):
            final_out.append(self.classifiers[i](outputs[i][:,0,:]).unsqueeze(1)) # bsz * 1 * 2
        final_out_mat = torch.nn.functional.softmax(torch.cat(final_out, dim=1), dim=2) # bsz * num of exp * 2
        gate_prob = torch.softmax(origin_gates, dim=-1)
        shifted_gate_prob = torch.softmax(shifted_gates, dim=-1)
        final_out_logits = torch.bmm(gate_prob.unsqueeze(1), final_out_mat).squeeze(1)
        shifted_final_logits = torch.bmm(shifted_gate_prob.unsqueeze(1), final_out_mat).squeeze(1)
        loss = []
        for logits in final_out:
            loss.append(self.cross_entropy(logits.squeeze(1), labels.view(-1)).view(-1,1))
        if len(loss) == 1:
            loss_mat = loss[0].view(-1,1)
        else:
            loss_mat = torch.cat(loss, dim=1) # bsz * # of experts
        final_loss = torch.mean(torch.sum(gate_prob * loss_mat, dim=1))
        
        return final_out, final_loss, shifted_final_logits, shifted_gates

class RobertaMoEModel_adaptive(RobertaPreTrainedModel):

    _keys_to_ignore_on_load_missing = [r"position_ids"]

    # Copied from transformers.models.bert.modeling_bert.BertModel.__init__ with Bert->Roberta
    def __init__(self, config, num_public_layers=12, num_experts=8, num_gate_layer=2):
        super().__init__(config)
        self.config = config
        self.encoder = RobertaMoEEncoder_adaptive(config, num_public_layers=num_public_layers, num_experts=num_experts, num_gate_layer=num_gate_layer)
        self.embeddings = RobertaEmbeddings(config)

    def get_input_embeddings(self):
        return self.embeddings.word_embeddings

    def set_input_embeddings(self, value):
        self.embeddings.word_embeddings = value

    def _prune_heads(self, heads_to_prune):
        """
        Prunes heads of the model. heads_to_prune: dict of {layer_num: list of heads to prune in this layer} See base
        class PreTrainedModel
        """
        for layer, heads in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(heads)

    # Copied from transformers.models.bert.modeling_bert.BertModel.forward
    
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        past_key_values=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None
    ):
        r"""
        encoder_hidden_states  (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length, hidden_size)`, `optional`):
            Sequence of hidden-states at the output of the last layer of the encoder. Used in the cross-attention if
            the model is configured as a decoder.
        encoder_attention_mask (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
            Mask to avoid performing attention on the padding token indices of the encoder input. This mask is used in
            the cross-attention if the model is configured as a decoder. Mask values selected in ``[0, 1]``: ``1`` for
            tokens that are NOT MASKED, ``0`` for MASKED tokens.
        past_key_values (:obj:`tuple(tuple(torch.FloatTensor))` of length :obj:`config.n_layers` with each tuple having 4 tensors of shape :obj:`(batch_size, num_heads, sequence_length - 1, embed_size_per_head)`):
            Contains precomputed key and value hidden states of the attention blocks. Can be used to speed up decoding.

            If :obj:`past_key_values` are used, the user can optionally input only the last :obj:`decoder_input_ids`
            (those that don't have their past key value states given to this model) of shape :obj:`(batch_size, 1)`
            instead of all :obj:`decoder_input_ids` of shape :obj:`(batch_size, sequence_length)`.
        use_cache (:obj:`bool`, `optional`):
            If set to :obj:`True`, :obj:`past_key_values` key value states are returned and can be used to speed up
            decoding (see :obj:`past_key_values`).
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        # return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        if not self.config.is_decoder:
            use_cache = False

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
            batch_size, seq_length = input_shape
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
            batch_size, seq_length = input_shape
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        device = input_ids.device if input_ids is not None else inputs_embeds.device

        # past_key_values_length
        past_key_values_length = past_key_values[0][0].shape[2] if past_key_values is not None else 0

        if attention_mask is None:
            attention_mask = torch.ones(((batch_size, seq_length + past_key_values_length)), device=device)
        if token_type_ids is None:
            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)

        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        extended_attention_mask: torch.Tensor = self.get_extended_attention_mask(attention_mask, input_shape, device)

        # If a 2D or 3D attention mask is provided for the cross-attention
        # we need to make broadcastable to [batch_size, num_heads, seq_length, seq_length]
        if self.config.is_decoder and encoder_hidden_states is not None:
            encoder_batch_size, encoder_sequence_length, _ = encoder_hidden_states.size()
            encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)
            if encoder_attention_mask is None:
                encoder_attention_mask = torch.ones(encoder_hidden_shape, device=device)
            encoder_extended_attention_mask = self.invert_attention_mask(encoder_attention_mask)
        else:
            encoder_extended_attention_mask = None

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
        # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)

        embedding_output = self.embeddings(
            input_ids=input_ids,
            position_ids=position_ids,
            token_type_ids=token_type_ids,
            inputs_embeds=inputs_embeds,
            past_key_values_length=past_key_values_length,
        )
        encoder_outputs, gates, shifted_gates = self.encoder(
            embedding_output,
            attention_mask=extended_attention_mask,
            head_mask=head_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_extended_attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states
        )
        

        return encoder_outputs, gates, shifted_gates

class RobertaMoEEncoder_adaptive(nn.Module):
    def __init__(self, config, num_public_layers, num_experts, num_gate_layer=2):
        super(RobertaMoEEncoder_adaptive,self).__init__()
        self.config = config
        self.layer = nn.ModuleList([RobertaLayer(config) for _ in range(num_public_layers)])
        self.expert_layer = nn.ModuleList([nn.ModuleList([RobertaLayer(config) for _ in range(config.num_hidden_layers-num_public_layers)]) for _ in range(num_experts)])
        self.num_gate_layer = num_gate_layer
        self.gate_layer = nn.ModuleList([RobertaLayer(config) for _ in range(self.num_gate_layer)])
        self.shift_layer = nn.ModuleList([RobertaLayer(config) for _ in range(self.num_gate_layer)])
        self.gate_classifier = RobertaClassificationHead_without_log(1024, config.hidden_dropout_prob, num_experts)
        self.shift_classifier = RobertaClassificationHead_without_log(1024, config.hidden_dropout_prob, num_experts)
    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        past_key_values=None,
        use_cache=None,
        output_attentions=False,
        output_hidden_states=False
    ):
        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None
        all_cross_attentions = () if output_attentions and self.config.add_cross_attention else None

        next_decoder_cache = () if use_cache else None
        for i, layer_module in enumerate(self.layer):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            layer_head_mask = head_mask[i] if head_mask is not None else None
            past_key_value = past_key_values[i] if past_key_values is not None else None
            if getattr(self.config, "gradient_checkpointing", False):

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return module(*inputs, past_key_value, output_attentions)

                    return custom_forward

                layer_outputs = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(layer_module),
                    hidden_states,
                    attention_mask,
                    layer_head_mask,
                    encoder_hidden_states,
                    encoder_attention_mask,
                )
            else:
                layer_outputs = layer_module(
                    hidden_states,
                    attention_mask,
                    layer_head_mask,
                    encoder_hidden_states,
                    encoder_attention_mask,
                    past_key_value,
                    output_attentions,
                )

            hidden_states = layer_outputs[0]
            if use_cache:
                next_decoder_cache += (layer_outputs[-1],)
            if output_attentions:
                all_self_attentions = all_self_attentions + (layer_outputs[1],)
                if self.config.add_cross_attention:
                    all_cross_attentions = all_cross_attentions + (layer_outputs[2],)
        hidden_states_inter = torch.clone(hidden_states)
        for i, layer_module in enumerate(self.gate_layer):
            if getattr(self.config, "gradient_checkpointing", False):

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return module(*inputs, past_key_value, output_attentions)

                    return custom_forward

                layer_outputs = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(layer_module),
                    hidden_states,
                    attention_mask,
                    None,
                    encoder_hidden_states,
                    encoder_attention_mask,
                )
            else:
                layer_outputs = layer_module(
                        hidden_states,
                        attention_mask,
                        None,
                        encoder_hidden_states,
                        encoder_attention_mask,
                        None,
                        output_attentions,
                )
            hidden_states = layer_outputs[0]
        gates = self.gate_classifier(hidden_states[:,0,:])
        # gate_prob = self.softmax(gates/gate_T) # bsz * # of experts
        # shift bias generation:
        hidden_states = hidden_states_inter
        for i, layer_module in enumerate(self.shift_layer):
            if getattr(self.config, "gradient_checkpointing", False):

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return module(*inputs, past_key_value, output_attentions)

                    return custom_forward

                layer_outputs = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(layer_module),
                    hidden_states,
                    attention_mask,
                    None,
                    encoder_hidden_states,
                    encoder_attention_mask,
                )
            else:
                layer_outputs = layer_module(
                        hidden_states,
                        attention_mask,
                        None,
                        encoder_hidden_states,
                        encoder_attention_mask,
                        None,
                        output_attentions,
                )
            hidden_states = layer_outputs[0]
        shift = self.shift_classifier(hidden_states[:,0,:]) # bsz , num of experts
        shifted_gates = shift + gates.detach()  # detach the gates from the calculation graph
        hidden_states_list = []
        
        for j, expert_module in enumerate(self.expert_layer):
            for i, layer_module in enumerate(expert_module):
                if i==0:
                    hidden_states = hidden_states_inter
                else:
                    hidden_states = hidden_states_list[j]
                if output_hidden_states:
                    all_hidden_states = all_hidden_states + (hidden_states,)

                layer_head_mask = head_mask[i] if head_mask is not None else None
                past_key_value = past_key_values[i] if past_key_values is not None else None
                if getattr(self.config, "gradient_checkpointing", False):

                    def create_custom_forward(module):
                        def custom_forward(*inputs):
                            return module(*inputs, past_key_value, output_attentions)

                        return custom_forward

                    layer_outputs = torch.utils.checkpoint.checkpoint(
                        create_custom_forward(layer_module),
                        hidden_states,
                        attention_mask,
                        layer_head_mask,
                        encoder_hidden_states,
                        encoder_attention_mask,
                    )
                else:
                    layer_outputs = layer_module(
                        hidden_states,
                        attention_mask,
                        layer_head_mask,
                        encoder_hidden_states,
                        encoder_attention_mask,
                        past_key_value,
                        output_attentions,
                    )

                hidden_states = layer_outputs[0]
                if i == 0:
                    hidden_states_list.append(hidden_states)
                else:
                    hidden_states_list[j] = hidden_states
                if use_cache:
                    next_decoder_cache += (layer_outputs[-1],)
                if output_attentions:
                    all_self_attentions = all_self_attentions + (layer_outputs[1],)
                    if self.config.add_cross_attention:
                        all_cross_attentions = all_cross_attentions + (layer_outputs[2],)
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        return hidden_states_list, gates, shifted_gates