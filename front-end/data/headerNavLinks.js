const headerNavLinks = [
  { href: '/', title: '主页' },
  { href: '/about', title: '关于我' },
  { href: '/posts', title: '博客' },
  { href: '/projects', title: '项目' },
  {
    type: 'dropdown',
    title: '其他',
    links: [
      { href: '/tags', title: '标签' },
      { href: '/tools', title: '工具' },
    ],
  },
]

export default headerNavLinks
