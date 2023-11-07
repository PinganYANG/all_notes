#### EMD+LSTM

- Utiliser LSTM pour prédire la quantité de retour des palettes dans les points relais. Ces palettes viennent des différents clients et les points relais se situent en toutes la france. 

##### Challenge

- Il y a beaucoup de points de relais en France
- La quantité de retour dans un point relais se varie beaucoup, par exemple, dans un petit point lyon, au janvier 15 il y a un retour de 17 palettes, le prochain retour est  au février 8 avec 9 palette, et le retour suivant peut au février 10 avec 20 palettes. 
  - Pour un point relais, la quantité de retour de palettes varie beaucoup et il n'y a pas assez des données pour analyser 



##### Solution

1. Utiliser la méthode clustering (K-means DBSCAN) pour merger les points relais pour prédire une tendance. Et car l'entreprise va recueillir toutes les palettes dans les petits points.

2.  La quantité varie beaucoup même après le clustering, ce qui conduit un problème pour LSTM d'utiliser la quantité de jour-1 comment la prédiction de ce jour. J'ai déjà utilisé la différence entre deux jour pour réduire le self-corrélation. Mais cela ne m'aide pas. J'ai vu des papers et j'ai trouvé il y a qqn utiliser un EMD (empirical mode decomposition) pour réduire ce self-corrélation. Je l'ai ajouté dans mon modèle de LLM et cela marche bien pour prédire la quantité.

   

##### Résultat

Le résultat est très intéressant pour l'entreprise et notre tuteur à l'école.



##### A faire

 Il n'a y pas assez des données pour bien prédire avec un Méthode de DL pour tous les points

La prédiction marche bien sur J+1, mais cela ne marche pas pour l'entreprise. Il faut la prédiction au moins de 30 jours pour planifier. 



#### Similarité de produits

Les produits de sons, genre musique et podcast

##### Challenge

Beaucoup des données et pas bien structuré

##### Solution

Après la nettoyage avec python, nous essayons deux méthodes pour trouver la similarité.

- une est density based, nous trouvons que DBSCAN est la meilleure. 

- L'autre est inspiré par spectral clustering. nous établissons une matrice de produit fois produit avec la quantité de client qui utilise ces deux produits en même temps. La distance entre deux produits est l'inverse de la quantité de client. 



#### Stage 1

Le but de pricing est de proposer un prix pertinent pour l'entreprise et pour ces clients. Un bon prix peut attirer les clients et augmenter le profit, c'est un gagnant-gagnant. Et mon rôle est de s'occuper tous les parties sans le front end dans ce projet. Premièrement c'est de créer un Data Pipeline pour nettoyer et traiter les données en utilisant les méthodes de pricing de l'entreprise. Et mettre ce pipeline en production avec Airflow. Cette partie est faite avec python.

Ensuite je m'occupe de construire des APIs entre frontend et backend, et aussi entre le pipeline de python et la base de données avec FastAPI et SQLAlchemy.

Suite à la besoin de client, j'ai établir une méthode de CDC change data capture pour mettre à jours des données en temps réel basé sur le Logical replication chez postgresql. 

Et j'ai aidé l'entreprise à améliorer la package interne de l'entreprise.



#### Stage 2

##### CV

J'ai établi un algo de détecter des anomalies sur les isolateurs des pylônes dans des images prises par la drone. 

##### Challenge

Les isolateurs dans des images sont très petites, et le background varie beaucoup. Mais les anomalies sont aussi très petites.

##### Solution

Pour trouver les petites anomalies  dans des images, il faut bien segmenter les isolateurs avec des anomalies. Et pour tourner dans un environnement industriel, il faut une méthode de détection robuste et bien testé. 

Premièrement nous utilisons YOLO pour localiser les isolateurs. Et ensuite nous utilisons une méthode de Segment Anything Model de Meta, qui peut segmenter directement les isolateurs depuis les bbox dans la partie avant. Et parce que les anomalies apparait pas mal autour du contour des isolateurs, nous ajoutons une partie d'utilisant Alpha Matting pour bien préciser le contour. Pour et après, nous entraînons un FastFlow Modèle pour bien détecter des anomalies.



![image-20231024154033583](C:\Users\cimum\AppData\Roaming\Typora\typora-user-images\image-20231024154033583.png)

##### LLM

Suite la tendance d'utiliser IA générative, j'ai aidé l'entreprise à créer un cadre de chatbot basé sur la base de connaissance personnalisée.

![image-20231024155630322](C:\Users\cimum\AppData\Roaming\Typora\typora-user-images\image-20231024155630322.png)

然后这个框架经过了一些适配用在了frhta网站的ChatBot上。