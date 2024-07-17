import pandas as pd
from pykeen.triples import TriplesFactory
import pykeen
from pykeen.pipeline import pipeline
import sys


def get_learned_embeddings(model):
    entity_representation_modules: List['pykeen.nn.RepresentationModule'] = model.entity_representations
    relation_representation_modules: List['pykeen.nn.RepresentationModule'] = model.relation_representations

    entity_embeddings: pykeen.nn.Embedding = entity_representation_modules[0]
    relation_embeddings: pykeen.nn.Embedding = relation_representation_modules[0]

    entity_embedding_tensor: torch.FloatTensor = entity_embeddings()
    relation_embedding_tensor: torch.FloatTensor = relation_embeddings()
    return entity_embedding_tensor, relation_embedding_tensor


def dataframe_embedding_donors(entity_embedding_tensor, entity, training):
    df = pd.DataFrame(entity_embedding_tensor.cpu().detach().numpy())
    df['ClinicalRecord'] = list(training.entity_to_id)
    new_df = df.loc[df.ClinicalRecord.isin(list(entity))]
    return new_df.iloc[:, :-1], new_df, df



def load_network_from_nt_file(name):
    kg = pd.read_csv(name, delimiter=",")
    tf_data = TriplesFactory.from_labeled_triples(triples=kg.to_numpy())
    donor = list(kg.loc[kg.p == 'p4-lucat:hasGender'].s.unique())
    return tf_data, donor


def create_model(tf_training, tf_testing, embedding, n_epoch, path):
    results = pipeline(
        training=tf_training,
        testing=tf_testing,
        model=embedding,
        training_loop='sLCWA',
        #         negative_sampler='bernoulli',
        negative_sampler_kwargs=dict(
            filtered=True,
        ),
        # model_kwargs=dict(embedding_dim=154),
        # Training configuration
        training_kwargs=dict(
            num_epochs=n_epoch,
            use_tqdm_batch=False,
        ),
        # training_batch_size=32,
        # Runtime configuration
        random_seed=1235,
        device='gpu',
    )
    model = results.model
    results.save_to_directory(path + embedding)
    return model, results


def main(*args):
    """Load TriplesFactory"""
    tf_data, donor = load_network_from_nt_file(args[1])
    """Split dataset in training and test set"""
    training, testing = tf_data.split(random_state=1234)
    """Build Knowledge Graph Embedding Model"""
    model_list = ['DistMult', 'TransE', 'TransH', 'ERMLP', 'RESCAL']
    # model_list = ['DistMult', 'TransE', 'TransH']
    for m in model_list:
        model, results = create_model(training, testing, m, 200, args[0])
        """Obtain the embeddings of entities and relations"""
        entity_embedding_tensor, relation_embedding_tensor = get_learned_embeddings(model)
        """Save the embeddings of donor entities"""
        df_donor, new_df, df_g1 = dataframe_embedding_donors(entity_embedding_tensor, donor, tf_data)
        new_df.to_csv(args[0]+m+'/embedding_donors.csv', index=None)


if __name__ == '__main__':
    main(*sys.argv[1:])
