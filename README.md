[//]: # (![logo]&#40;logo.png&#41;)
# WhisprRec :panda_face:
WhisprRec is a collection of deep learning applications in recommendation algorithms. 
It mainly collects the mainstream recommendation models and our team's research results in recommendation systems.
The framework uses pytorch.
IF you find this project is valuable, please **star** :).

Thanks to the work of Rechorus and Recbole.

## Get Started :snail:
1. You have to install and configure the following environment: (My environment)
- Python >= 3.8
- Pytorch = 1.9.0
- Pandas = 1.4.4
- Numpy = 1.23.2
- tqdm = 4.51.0

2. Clone this repository:
```bash
git clone https://github.com/HeyWeCome/WhisprRec.git
```

3. cd into 'src'
```bash
cd WhisprRec/src
```

4. Run your or build-in dataset
```bash
python main.py --model_name BPRMF --emb_size 64 --lr 1e-3 --l2 1e-6 --dataset ml-100k
```

## The Current Model List :owl:
In ml-100k dataset.

| ****Model****      | ****HR@10**** | ****NDCG@10**** | ****Best hyper-parameters****                                                 | ****Description**** |
|--------------------|---------------|-----------------|-------------------------------------------------------------------------------|---------------------|
| ****General****    | ============  | ============    | ============                                                                  | ============        |
| ****BUIR****       | 0.1114        | 0.0626          | lr=5e-4                                                                       | SIGIR'21            |
| ****BPRMF****      | 0.2254        | 0.1085          | lr=1e-3                                                                       | UAI'09              |
| ****LightGCN****   | 0.2292        | 0.1174          | lr=2e-3, gcn_layers=2                                                         | SIGIR'20            |
| ****SGL****        | 0.2287        | 0.1187          | lr=2e-3, gcn_layers=2, ssl_tau=0.1, drop_ratio=0.1, type=ED, ssl_weight=0.001 | SIGIR’21            |
| ****Sequential**** | ============  | ============    | ============                                                                  | ============        |
| ****SASRec****     | 0.3166        | 0.1798          | lr=5e-4, sample=LS, num_layers=1, num_heads=4                                 | ICDM‘18             |
