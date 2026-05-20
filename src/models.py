import torch
import torch.nn as nn

class NCF(nn.Module):
    """
    Neural Collaborative Filtering (NeuMF) - He et al. 2017
    Combine GMF (Generalized Matrix Factorization) et MLP.
    """
    def __init__(self, num_users: int, num_items: int, 
                 embed_dim: int = 64, mlp_layers: list = [128, 64, 32, 16],
                 dropout: float = 0.2):
        super(NCF, self).__init__()
        
        self.num_users = num_users
        self.num_items = num_items
        
        # GMF Embeddings
        self.embedding_user_gmf = nn.Embedding(num_embeddings=num_users, embedding_dim=embed_dim)
        self.embedding_item_gmf = nn.Embedding(num_embeddings=num_items, embedding_dim=embed_dim)
        
        # MLP Embeddings
        self.embedding_user_mlp = nn.Embedding(num_embeddings=num_users, embedding_dim=embed_dim)
        self.embedding_item_mlp = nn.Embedding(num_embeddings=num_items, embedding_dim=embed_dim)
        
        # Initialisation
        nn.init.normal_(self.embedding_user_gmf.weight, std=0.01)
        nn.init.normal_(self.embedding_item_gmf.weight, std=0.01)
        nn.init.normal_(self.embedding_user_mlp.weight, std=0.01)
        nn.init.normal_(self.embedding_item_mlp.weight, std=0.01)
        
        # MLP Layers
        mlp_modules = []
        in_dim = embed_dim * 2 # user_embed + item_embed concaténés
        
        for out_dim in mlp_layers:
            mlp_modules.append(nn.Linear(in_dim, out_dim))
            mlp_modules.append(nn.ReLU())
            mlp_modules.append(nn.Dropout(p=dropout))
            in_dim = out_dim
            
        self.mlp = nn.Sequential(*mlp_modules)
        
        # Output Layer (Fusion GMF + MLP)
        self.output_layer = nn.Linear(embed_dim + mlp_layers[-1], 1)
        nn.init.kaiming_uniform_(self.output_layer.weight, a=1, nonlinearity='sigmoid')
        
    def forward(self, user_indices: torch.Tensor, item_indices: torch.Tensor) -> torch.Tensor:
        # GMF part
        user_embedding_gmf = self.embedding_user_gmf(user_indices)
        item_embedding_gmf = self.embedding_item_gmf(item_indices)
        gmf_vector = torch.mul(user_embedding_gmf, item_embedding_gmf) # Hadamard product
        
        # MLP part
        user_embedding_mlp = self.embedding_user_mlp(user_indices)
        item_embedding_mlp = self.embedding_item_mlp(item_indices)
        mlp_vector = torch.cat([user_embedding_mlp, item_embedding_mlp], dim=-1)
        mlp_vector = self.mlp(mlp_vector)
        
        # Concatenate GMF and MLP vectors
        concat_vector = torch.cat([gmf_vector, mlp_vector], dim=-1)
        
        # Output
        logits = self.output_layer(concat_vector)
        return logits.squeeze(-1)

class ContextEncoder(nn.Module):
    """
    Encodeur des features contextuelles.
    """
    def __init__(self, context_dim: int, embed_dim: int = 32):
        super(ContextEncoder, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(context_dim, embed_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(embed_dim * 2, embed_dim),
            nn.ReLU()
        )
        
    def forward(self, context_features: torch.Tensor) -> torch.Tensor:
        return self.net(context_features)

class NCFContext(nn.Module):
    """
    NCF avec intégration du contexte.
    Fusion_type: 'concat', 'film', ou 'attention'.
    """
    def __init__(self, num_users: int, num_items: int, context_dim: int,
                 embed_dim: int = 64, mlp_layers: list = [128, 64, 32, 16],
                 context_embed_dim: int = 32,
                 dropout: float = 0.2, fusion_type: str = 'concat'):
        super(NCFContext, self).__init__()
        
        self.fusion_type = fusion_type
        
        # NCF Core
        self.ncf_core = NCF(num_users, num_items, embed_dim, mlp_layers, dropout)
        self.ncf_output_dim = embed_dim + mlp_layers[-1]
        
        # Context Encoder
        self.context_encoder = ContextEncoder(context_dim, context_embed_dim)
        
        # Stratégies de fusion
        if fusion_type == 'concat':
            self.final_mlp = nn.Sequential(
                nn.Linear(self.ncf_output_dim + context_embed_dim, 32),
                nn.ReLU(),
                nn.Linear(32, 1)
            )
        elif fusion_type == 'film':
            self.film_gamma = nn.Linear(context_embed_dim, self.ncf_output_dim)
            self.film_beta = nn.Linear(context_embed_dim, self.ncf_output_dim)
            self.final_mlp = nn.Linear(self.ncf_output_dim, 1)
        elif fusion_type == 'attention':
            self.attn = nn.Sequential(
                nn.Linear(self.ncf_output_dim + context_embed_dim, 1),
                nn.Sigmoid()
            )
            self.final_mlp = nn.Sequential(
                nn.Linear(self.ncf_output_dim + context_embed_dim, 32),
                nn.ReLU(),
                nn.Linear(32, 1)
            )
        else:
            raise ValueError(f"Fusion type {fusion_type} inconnu.")

    def forward(self, user_indices: torch.Tensor, item_indices: torch.Tensor, 
                context_features: torch.Tensor) -> torch.Tensor:
        
        # 1. NCF Repr
        user_embedding_gmf = self.ncf_core.embedding_user_gmf(user_indices)
        item_embedding_gmf = self.ncf_core.embedding_item_gmf(item_indices)
        gmf_vector = torch.mul(user_embedding_gmf, item_embedding_gmf)
        
        user_embedding_mlp = self.ncf_core.embedding_user_mlp(user_indices)
        item_embedding_mlp = self.ncf_core.embedding_item_mlp(item_indices)
        mlp_vector = torch.cat([user_embedding_mlp, item_embedding_mlp], dim=-1)
        mlp_vector = self.ncf_core.mlp(mlp_vector)
        
        ncf_repr = torch.cat([gmf_vector, mlp_vector], dim=-1)
        
        # 2. Context Repr
        context_repr = self.context_encoder(context_features)
        
        # 3. Fusion
        if self.fusion_type == 'concat':
            fused = torch.cat([ncf_repr, context_repr], dim=-1)
            logits = self.final_mlp(fused)
        elif self.fusion_type == 'film':
            gamma = self.film_gamma(context_repr)
            beta = self.film_beta(context_repr)
            fused = ncf_repr * (1 + gamma) + beta
            logits = self.final_mlp(fused)
        elif self.fusion_type == 'attention':
            concat_repr = torch.cat([ncf_repr, context_repr], dim=-1)
            attn_weights = self.attn(concat_repr)
            fused = concat_repr * attn_weights
            logits = self.final_mlp(fused)
            
        return logits.squeeze(-1)