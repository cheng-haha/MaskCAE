'''
Description: 
Date: 2024-08-13 13:08:07
LastEditTime: 2024-08-13 19:09:44
FilePath: /chengdongzhou/action/MaskCAE/common.py
'''

channel_list    =  {
                    'ucihar':   [ 16, 32,  64,  128, 6   ],
                    'motion':   [ 16, 32,  64,  128, 6   ],
                    'uschad':   [ 16, 32,  64,  128, 12  ],
                        }

conv_list       =   {
                    'ucihar':   [ (5,1), (1,1), (2,0) ],
                    'motion':   [ (5,1), (1,1), (2,0) ],
                    'uschad':   [ (5,1), (1,1), (2,0) ],
        
                        }
    
maxp_list       =   {
                    'ucihar':   [ (5,1), (2,1) , (2,0) ],
                    'motion':   [ (5,1), (2,1) , (2,0) ],
                    'uschad':   [ (5,1), (2,1) , (2,0) ],
                        }

first_maxp_list   =   {
                    'ucihar': [ (5,1), (4,1), (2,0) ],
                    'uschad': [ (5,1), (4,1), (2,0) ],
                    'motion': [ (5,1), (4,1), (2,0) ],
                        }



