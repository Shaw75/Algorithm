#include <iostream>
#include<vector>
#include <map>
#include <algorithm>
#include<list>
#include<stdio.h>

#include <stdlib.h>
#include <fstream>
#include <numeric>
#include <set>
#include <queue>
#include <stack>

using namespace std;


//下标之和
/*
给定一个整数数组 nums ，对其从左到右扫描，并返回一个数组，数组中的第 i 个元素为 nums 中第 i 个整数最近 5 次出现的下标之和，如果该整数在此之前出现次数不满 5 次则为 -1。

示例 1:

输入：
nums = [1,1,1,1,1,1]

输出：[-1,-1,-1,-1,-1,10]

解释：
只有 nums 中的第 6 个出现的 1 ，满足上述条件，故返回的数组中最后一个元素等于 0+1+2+3+4 = 10。

提示：
6 <= nums.length <= 10^5
-10^4 <= nums[i] <= 10^4
*/
//思路： 长度5队列组成哈希表
class Solution2 {
public:
    vector<int> solve(vector<int>& nums) {
        map<int,vector<int>> map;
        int n = nums.size();
        vector<int> res(n,0);
          for(int i = 0; i<n;++i){
              if(map.count(nums[i])){
                    if(map[nums[i]].size()<5){
                        res[i]=-1;
                    }else{
                        for(int j=(int)(map[nums[i]].size()-1);j>=(int)(map[nums[i]].size()-5);--j)
                            res[i]+=map[nums[i]][j];
                    }
                 map[nums[i]].push_back(i);
              }else{
                  res[i]=-1;
                  vector<int>temp;
                  temp.push_back(i);
                  map[nums[i]]=temp;

              }
          }
        return res;
    }

};




//力扣第15题. 三数之和
class Solution3 {
public:
    vector<vector<int>> threeSum(vector<int>& nums) {
        int size = nums.size();
        if(size <3) return {};
        vector<vector<int>> res;
        sort(nums.begin(),nums.end());
        for (int i = 0; i < nums.size(); ++i) {
            if(nums[i]>0)  return res;
            if(i>0  && nums[i]==nums[i-1]) continue;
            int left =i+1;
            int right=size-1;
            while(left<right){
                if(left<right && nums[left]+nums[right]<-nums[i]) left++;
                else if(left<right && nums[left]+nums[right]>-nums[i]) right--;
                else {
                    res.push_back(vector<int>{nums[i],nums[left],nums[right]});
                    left++;
                    right--;
                    while (left < right && nums[left] == nums[left-1])  left++;
                    while (left < right && nums[right] == nums[right+1])    right--;
                }
            }

        }
        return  res;
    }
};

//快速排序
class Solution4{
  public:
     int pivot(vector<int> &nums,int left,int right){
         int i=left ,j=right,mid=nums[left];
         while(i<j){
             while(i<j && nums[j]>mid) j--;
             if(i<j) {
                 swap(nums[i],nums[j]);
                 i++;
             }
             while(i<j&&nums[i]<mid) i++;
             if(i<j){
                 swap(nums[i],nums[j]);
                 j--;
             }

         }
         return i;

     }
    vector<int> sortArray(vector<int> &nums,int left,int right){
         int mid;
         if(left<right){
             mid= pivot(nums,left,right);
             sortArray(nums,left,mid-1);
             sortArray(nums,mid+1,right);

         }
        return  nums;
     }

};

//只出现过一次的数字
class Solution5{
public:


        int singleNumber(vector<int>& nums) {
            int res=nums[0];
           for(int i=1;i<nums.size();++i){
               res ^=nums[i];
        }
            return res;
    };



};
//删除有序数组中的重复项
class Solution6 {
public:
    int removeDuplicates(vector<int>& nums) {
        int n =nums.size();
        if(n==0) return 0;
        int fast=1,slow=1;
         while(fast<n){
             if(nums[fast]!=nums[fast-1]){
                   nums[slow]=nums[fast];
                   ++slow;
             }
             ++fast;
        }
        return slow;
    }
};

//删除链表倒数第N个节点
class Solution7{
public:
    class Solution {

    public:
        struct ListNode {
            int val;
            ListNode *next;
            ListNode() : val(0), next(nullptr) {}
            ListNode(int x) : val(x), next(nullptr) {}
            ListNode(int x, ListNode *next) : val(x), next(next) {}
             };
        ListNode* removeNthFromEnd(ListNode* head, int n) {
            if(head== nullptr) return 0;
            auto fast=head;
            auto slow =head;
            while(n>0){
                fast=fast->next;
                --n;
        }
            if(fast== nullptr) return head->next;
            while(fast->next!= nullptr){
                fast=fast->next;
                slow=slow->next;
            }
            slow->next=slow->next->next;
            return head;

    };
    };

};


//跳跃游戏2
class Solution8{
public:
    int jump(vector<int>& nums) {
        int max=0;
        int n=nums.size();
        int end=0;
        int step=0;
        for(int i=0;i<n-1;++i){
            max=std::max(max,i+nums[i]);
            if(i==end){
                end=max;
                ++step;

            }

        }
        return step;
    }

};
//旋转矩阵
class Solution {
public:

    void rotate(vector<vector<int>>& matrix) {
        
       int n= matrix.size();
       for(int i=0;i<n;++i){
           for(int j=i+1;j<n;++j){
               swap(matrix[i][j],matrix[j][i]);

           }
       }
       for(int i=0;i<n;++i){
           for(int j=0;j<n/2;++j)
               swap(matrix[i][j],matrix[i][n-j-1]);
       }

    }
};


//环形链表2

class Solution10 {
public:
    ListNode *detectCycle(ListNode *head) {
        ListNode* p = head;
        ListNode* q = head;
        while (q && q->next) {
            p = p->next;
            q = q->next->next;
            if (p == q)
                break;
        }
        if (!q || !q->next)
            return NULL;
        p = head;
        while (p != q) {
            p = p->next;
            q = q->next;
        }
        return p;
    }
};

//
class Solution11 {
public:
    int findKthLargest(vector<int>& nums, int k) {
        std::sort(nums.begin(), nums.end());
        return nums[nums.size()-k];
      }
};

//二叉查找树
template <class K>
struct BStreeNode {
    BStreeNode* lchild;
    BStreeNode* rchild;
    K key;
    BStreeNode():lchild(nullptr),rchild(nullptr),K(key){

    }

};


//卡牌分组
class Solution12 {
public:
    bool hasGroupsSizeX(vector<int>& deck) {
        //分析题意：数组中的每个数字出现的次数是要进行统计的
        //我们可以想到 11 11 22 33 这种分配方案也是合理的
        //X >= 2那么这个划分后的数组长度至少是2，因此可以快速想到统计次数，求次数的最大公约数，然后判断其是否 >=2
        unordered_map<int,int> number_map;
        for (int num : deck){
            number_map[num]++;
        }
        int gcdnumber = number_map.begin()->second;
        for (auto &iter : number_map){
            gcdnumber = gcd(iter.second,gcdnumber);
        }
        return gcdnumber >= 2;
    }
};

//第三大的数
class Solution13 {
public:
    int thirdMax(vector<int>& nums) {
          int n= nums.size();
        sort(nums.begin(),nums.end(),greater<int>());
         for(int i=1,j=1;i<n;++i){
             if(nums[i-1]!=nums[i] && ++j==3)
                 return nums[i];

         }
        return nums[0];

    }

};
//
class Solution14 {
public:
    int thirdMax(vector<int>& nums) {
       set<int> set;
       for(int num:nums){
           set.insert(num);
           if(set.size()>3)  set.erase(set.begin());
       }
        return set.size()==3?*set.begin():*set.rbegin();
    }

};

//斐波拉契链表
class Solution15 {
    struct ListNode {
        int val;
       ListNode *next;
       ListNode(){

       }
        ListNode(int x) : val(x), next(NULL) {}
         };
public:
    ListNode* solve(ListNode* head) {
      ListNode* node1=new ListNode;
      ListNode* node2=node1;
      node2->next=head;
      node2=node2->next;
      int n1=1,n2=1;
      int count =0;
      while (head!= nullptr){
          ++count;
          if(n1+n2 == count){
              node2->next=head;
              node2=node2->next;
              n1=n2;
              n2=count;
          }
          head=head->next;

      }
        node2->next= nullptr;
        return node1->next;
    }
};

//移动零
class Solution16 {
public:
    void moveZeroes(vector<int>& nums) {
        int slow=0;
        int n= nums.size();
        for(int i=1;i<n;++i){
            if(nums[slow]!=0) slow++;
            if(nums[i]!=nums[i-1]){
                  swap(nums[i],nums[slow]);

            }

        }

    }
};

//数组大小减半
class Solution17 {
public:
    int minSetSize(vector<int>& arr) {
           int n=arr.size(),nums=1,res=0,all=0;
           int target = n/2;
           vector<int> v;
           sort(arr.begin(),arr.end());
        for (int i = 0; i < n-1; ++i) {
            if(arr[i]==arr[i+1]) nums++;
            else{
            v.push_back(nums);
            nums=1;
        }
        }
        v.push_back(nums);
        sort(v.begin(),v.end());
        for (int i = v.size()-1; i >=0 ; --i) {
            res++;
            all+=v[i];
            if(all>=target) return res;

        }
        return 0;

    }
};

//寻找右区间
class Solution18 {
public:
    vector<int> findRightInterval(vector<vector<int>>& intervals) {
         vector<pair<int ,int >>  start;
         int n =intervals.size();
        for (int i = 0; i < n; ++i) {
               start.emplace_back(intervals[i][0],i);
        }
        sort(start.begin(),start.end());
        vector<int> res(n,-1);

        for (int i = 0; i < n; ++i) {
            auto it = lower_bound(start.begin(),start.end(), make_pair(intervals[i][1],0));
            if(it!=start.end())  res[i]=it->second;
        }
        return res;
    }
};

//滑动窗口最大值
class Solution19 {
public:
    vector<int> maxSlidingWindow(vector<int>& nums, int k) {
        int n = nums.size();
        priority_queue<pair<int, int>> q;
        for (int i = 0; i < k; ++i) {
            q.emplace(nums[i], i);
        }
        vector<int> ans = {q.top().first};
        for (int i = k; i < n; ++i) {
            q.emplace(nums[i], i);
            while (q.top().second <= i - k) {
                q.pop();
            }
            ans.push_back(q.top().first);
        }
        return ans;
    }
};


//统计目标子序列
class Solution20{
public:
    const int MOD=1000000007;
    void ADD(int &a,int b) {
        a=(a+b)%MOD;
    }
    int solve(string tmp) {
        vector<int>res(5,0);
        for(auto &ch:tmp){
               if(ch=='a'){
                   ADD(res[0],1);
                   ADD(res[4],res[3]);
               }else if(ch=='b'){
                   ADD(res[1],res[0]);
                   ADD(res[3],res[2]);
               }else{
                   ADD(res[2],res[1]);
               }
        }
        return res[4];
    }
};


class Solution21 {
public:
    int canCompleteCircuit(vector<int>& gas, vector<int>& cost) {
        int n=gas.size();
        int totalGas=0;
        int totalCost=0;
        for (int i = 0; i < n; ++i) {
            totalGas+=gas[i];
            totalCost+=cost[i];
        }
        if(totalGas-totalCost<0)return -1;
        int currentGas=0;
        int start=0;
        for (int i = 0; i < n; ++i) {
            currentGas +=gas[i]-cost[i];
            if(currentGas<0) {
                currentGas = 0;
                start = i + i;
            }
        }
        return  start;
        }


};


//序列重排  (超时)
class Solution22 {
public:
    int solve(vector<int>& nums, int m, int kth) {
        queue<int> q;
        stack<int> s;
        int n = nums.size();
        while (m--) {
            for (int i = 0; i < n; ++i) {
                if (i % 2 == 0) {
                    s.push(nums[i]);
                } else if (i % 2 != 0) {
                    q.push(nums[i]);
                }
            }
            int j=0;
            while(!q.empty()){
                nums[j++] =  q.front();
                q.pop();
            }
            while(!s.empty()){
                nums[j++] = s.top();
                s.pop();
            }


        }
        return  nums[kth-1];
    }
};


//序列重排  (超时)
class Solution23 {
public:
    int reverse(int kth,int n){
        if(kth<n/2)  return 2*kth+1;
        if(kth>n/2)  return 2*(n-kth-1);
    }
    int solve(vector<int>& nums, int m, int kth) {
      int n=nums.size();
      while(m--){
          kth = reverse(kth,n);
      }
     return nums[kth];
    }
};


class Node {
public:
    int val;
    Node* next;
    Node* random;

    Node(int _val) {
        val = _val;
        next = NULL;
        random = NULL;
    }
};
class Solution24 {
public:
    Node* copyRandomList(Node* head) {

         if(head==nullptr)  return nullptr;
        Node * cur=head;
        unordered_map<Node*,Node*> map;
        while(cur!= nullptr){
            map[cur] = new Node(cur->val);
            cur=cur->next;
        }
        cur =head;
        while(cur!= nullptr){
            map[cur]->next =map[cur->next];
            map[cur]->random =map[cur->random];
            cur =cur->next;
        }
        return map[head];
    }
};


//侧视图
struct TreeNode {
     int val;
     TreeNode *left;
      TreeNode *right;
     TreeNode() : val(0), left(nullptr), right(nullptr) {}
       TreeNode(int x) : val(x), left(nullptr), right(nullptr) {}
      TreeNode(int x, TreeNode *left, TreeNode *right) : val(x), left(left), right(right) {}
};
class Solution25{
public:
    vector<int> rightSideView(TreeNode* root) {
        queue<TreeNode *> q;
        vector<int> res;
        if(root != nullptr) q.push(root);

        while(!q.empty()){
            int n = q.size();
            for(int i=0;i<n;++i){
                 TreeNode * node =q.front();
                 if(i==n-1) res.push_back(node->val);
                 q.pop();
                 if(node->left) q.push(node->left);
                 if(node->right) q.push(node->right);
            }
        }

     return res;
    }
};




//线性结构组合
class Solution26 {
public:
    int solve(vector<vector<int>>& types, vector<int>& nums, int kth) {
          //表示有n个数据结构
          int n =types.size();
          int stack_len=0;
        for (int i =0;i<n;++i){
            if(types[i][0]==0){
                stack_len += types[i][1]-1;
            }
        }
       return nums[stack_len+kth-1];
    }
};


class Solution27 {
public:
    int majorityElement(vector<int>& nums) {
      unordered_map<int,int> map;
       int n = nums.size();
       int cut=0;
       int max=0;
       for(int i=0;i<n;++i){
         map[nums[i]]++;
          if(map[nums[i]]>cut){
              max=nums[i];
              cut=map[nums[i]];
          }
       }
       return max;

    }
};

//序列化和反序列号
class Codec {
public:
    void rserialize(TreeNode *root,string &str){
         if(root== nullptr)
             str+="null,";
         else{
             str += to_string(root->val)+",";
             rserialize(root->left,str);
             rserialize(root->right,str);
         }
    }
    TreeNode* rdeserialize(string data) {

    }

    // Encodes a tree to a single string.
    string serialize(TreeNode* root) {
        string ret;
        rserialize(root, ret);
        return ret;
    }

    // Decodes your encoded data to tree.
    TreeNode* deserialize(string data) {

    }
};



//中序遍历
class Solution28 {
public:
    vector<int> res;
    //中序遍历
    vector<int> midOrder(TreeNode *root){
        if(root== nullptr) return res;
        midOrder(root->left);
        res.push_back(root->val);
        midOrder(root->right);

    }
    int kthSmallest(TreeNode* root, int k) {
        return res[k-1];
    }

};


//只出现一次的数字
class Solution29 {
public:
    vector<int> singleNumber(vector<int>& nums) {
         vector<int> res;
         unordered_map<int,int>map;
         int n = nums.size();
        for (int i=0;i<n;i++){
            map[nums[i]]++;
        }
        for(auto it=map.begin(); it!=map.end();it++){
                  if(it->second == 1) {
                      res.push_back(it->first);
                  }
        }
    return res;
    }
};



//左叶子子树之和
class Solution31 {
public:

    int sumOfLeftLeaves(TreeNode* root) {
        if(root== nullptr) return 0;
        int res=0;
        if(root->left!= nullptr && root->left->left== nullptr &&root->right->right== nullptr)
            res +=root->val;

        return sumOfLeftLeaves(root->left)+ sumOfLeftLeaves(root->right)+res;

    }
};


//最小化排队等待时间
/*
 有 n 个人在等待办理事务，其中第 i 个人的事务需要 w[i] 分钟完成。现在希望你安排他们办理事务的顺序，从而使得每个人的等待时间之和最小，并返回最小的排队等待时间总和。

示例 1：

输入：w = [1,3,2]

输出：4

解释：

首先安排第一个人办理，第一个人等待时间为 0。需要 1 分钟。
然后安排第三个人办理，第三个人等待第一个人等待了 1 分钟，他自己的业务办理需要 2 分钟。
最后安排第二个人办理，他已经等待了 3 分钟，他自身的业务办理需要 3 分钟。
第三个人等待了 1 分钟，第二个人等待了 3 分钟，因此总的等待时间是 4 分钟。这是最少的等待方案。

提示：

1 <= w.length <= 10^4
1 <= w[i] <= 100

 */
class Solution32 {
public:
    int minimumWaitingTime(vector<int>& w) {
        int n = w.size();
        std::sort(w.begin(), w.end());
        int res = 0;
        for (int i = 0; i < n; ++i) {
            res+= w[i]*(n-1-i);
        }
        return res;
    }
};



/*
  统计出现次数
给定一个有序数组 nums，以及一个目标数 target。返回数组 nums 中 target 的出现次数。

请你实现时间复杂度为 O(logn) 并且只使用 常数级别额外空间 的解决方案。

示例 1：

输入：nums = [1,1,2,2,2,2,3,3,5,6,7,7,7], target = 3

输出：2

提示：

1 <= nums.length <= 10^5
-10^5 <= nums[i], target <= 10^5
 */
class Solution30{
public:
    int countOccurrences(vector<int>& nums, int target) {
        return upper_bound(nums.begin(),nums.end(),target)-upper_bound(nums.begin(),nums.end(),target-1);
    }
};

